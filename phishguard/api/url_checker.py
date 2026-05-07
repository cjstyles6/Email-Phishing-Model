"""URL extraction, local reputation checks, and VirusTotal lookups."""

from __future__ import annotations

import base64
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


URL_REGEX = re.compile(
    r"(?P<url>https?://[^\s<>'\"]+|www\.[^\s<>'\"]+)",
    re.IGNORECASE,
)
IPV4_HOST_REGEX = re.compile(
    r"^(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}$"
)
URL_ENCODED_REGEX = re.compile(r"%[0-9a-fA-F]{2}")


@dataclass
class URLChecker:
    """Analyze URLs using local heuristics and optional VirusTotal checks."""

    api_key: str | None = None
    max_requests_per_minute: int = 4
    request_timeout_seconds: int = 15
    _request_timestamps: deque[float] = field(default_factory=deque)

    virustotal_base_url: str = "https://www.virustotal.com/api/v3"
    shortener_domains: frozenset[str] = frozenset(
        {
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "goo.gl",
            "ow.ly",
            "buff.ly",
            "short.link",
            "rb.gy",
            "cutt.ly",
        }
    )
    protected_brand_domains: dict[str, str] = field(
        default_factory=lambda: {
            "paypal": "paypal.com",
            "amazon": "amazon.com",
            "apple": "apple.com",
            "microsoft": "microsoft.com",
            "netflix": "netflix.com",
            "google": "google.com",
            "facebook": "facebook.com",
            "instagram": "instagram.com",
            "linkedin": "linkedin.com",
            "chase": "chase.com",
            "wellsfargo": "wellsfargo.com",
            "bankofamerica": "bankofamerica.com",
        }
    )

    def extract_urls(self, email_text: str) -> list[str]:
        """Extract http(s) URLs and www-prefixed bare domains from text."""
        seen: set[str] = set()
        urls: list[str] = []

        for match in URL_REGEX.finditer(email_text):
            url = match.group("url").rstrip(").,;!?:]")
            if url not in seen:
                seen.add(url)
                urls.append(url)

        return urls

    def check_email_text(self, email_text: str) -> dict[str, Any]:
        """Extract and check all URLs in an email body."""
        urls = self.extract_urls(email_text)
        results = [self.check_url(url) for url in urls]
        malicious_count = sum(1 for result in results if result["is_malicious"])
        suspicious_count = sum(1 for result in results if result["is_suspicious"])
        shortener_count = sum(1 for result in results if result["is_shortener"])

        overall_url_threat = "safe"
        if malicious_count:
            overall_url_threat = "danger"
        elif suspicious_count or shortener_count:
            overall_url_threat = "suspicious"

        return {
            "urls_found": len(urls),
            "results": results,
            "overall_url_threat": overall_url_threat,
            "malicious_count": malicious_count,
            "suspicious_count": suspicious_count,
            "shortener_count": shortener_count,
            "url_threat_score": self._overall_score(results),
        }

    def check_url(self, url: str) -> dict[str, Any]:
        """Check one URL using local signals plus VirusTotal when available."""
        normalized_url = self._normalize_url(url)
        parsed_url = urlparse(normalized_url)
        domain = self._normalize_domain(parsed_url.hostname or "")
        flags = self._local_flags(normalized_url, parsed_url, domain)
        virustotal_stats: dict[str, int] = {}

        if self.api_key:
            vt_stats, vt_flags = self._check_virustotal(normalized_url)
            virustotal_stats = vt_stats
            flags.extend(flag for flag in vt_flags if flag not in flags)
        else:
            flags.append("virustotal_disabled")

        malicious_count = virustotal_stats.get("malicious", 0)
        suspicious_count = virustotal_stats.get("suspicious", 0)
        is_malicious = malicious_count >= 2
        is_shortener = "url_shortener" in flags
        is_suspicious = (
            suspicious_count >= 2
            or is_shortener
            or any(
                flag
                in {
                    "ip_address_url",
                    "excessive_subdomains",
                    "misleading_brand_domain",
                    "non_standard_port",
                    "url_encoded_characters",
                    "virustotal_suspicious",
                }
                for flag in flags
            )
        )

        if is_malicious and "virustotal_malicious" not in flags:
            flags.append("virustotal_malicious")
        if suspicious_count >= 2 and "virustotal_suspicious" not in flags:
            flags.append("virustotal_suspicious")

        return {
            "url": url,
            "is_malicious": is_malicious,
            "is_suspicious": is_suspicious,
            "is_shortener": is_shortener,
            "virustotal_stats": virustotal_stats,
            "flags": flags,
            "risk_score": self._risk_score(flags, is_malicious, is_suspicious),
        }

    def _local_flags(self, url: str, parsed_url, domain: str) -> list[str]:
        """Run local URL pattern checks before any external reputation call."""
        flags: list[str] = []

        if domain in self.shortener_domains:
            flags.append("url_shortener")
        if IPV4_HOST_REGEX.match(domain):
            flags.append("ip_address_url")
        if self._has_excessive_subdomains(domain):
            flags.append("excessive_subdomains")
        if self._has_misleading_brand_domain(domain):
            flags.append("misleading_brand_domain")
        port = self._safe_port(parsed_url)
        if port and port not in {80, 443}:
            flags.append("non_standard_port")
        if URL_ENCODED_REGEX.search(url):
            flags.append("url_encoded_characters")

        return flags

    def _check_virustotal(self, url: str) -> tuple[dict[str, int], list[str]]:
        """Submit and retrieve VirusTotal URL stats without failing the scan."""
        flags: list[str] = []

        if not self._consume_request_slot():
            return {}, ["rate_limited"]

        try:
            self._post_json(
                f"{self.virustotal_base_url}/urls",
                data=urlencode({"url": url}).encode("utf-8"),
                content_type="application/x-www-form-urlencoded",
            )
        except Exception:
            return {}, ["virustotal_unreachable"]

        if not self._consume_request_slot():
            return {}, ["rate_limited"]

        try:
            url_id = self._virustotal_url_id(url)
            report = self._get_json(f"{self.virustotal_base_url}/urls/{url_id}")
            stats = (
                report.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
            )
            normalized_stats = self._normalize_stats(stats)

            if normalized_stats.get("malicious", 0) >= 2:
                flags.append("virustotal_malicious")
            if normalized_stats.get("suspicious", 0) >= 2:
                flags.append("virustotal_suspicious")

            return normalized_stats, flags
        except Exception:
            return {}, ["virustotal_unreachable"]

    def _consume_request_slot(self) -> bool:
        """Allow at most max_requests_per_minute VirusTotal requests."""
        now = time.monotonic()
        while self._request_timestamps and now - self._request_timestamps[0] >= 60:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self.max_requests_per_minute:
            return False

        self._request_timestamps.append(now)
        return True

    def _post_json(self, url: str, *, data: bytes, content_type: str) -> dict[str, Any]:
        request = Request(
            url,
            data=data,
            headers={
                "x-apikey": self.api_key or "",
                "Content-Type": content_type,
            },
            method="POST",
        )
        return self._open_json(request)

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"x-apikey": self.api_key or ""}, method="GET")
        return self._open_json(request)

    def _open_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("VirusTotal request failed") from exc

    @staticmethod
    def _normalize_url(url: str) -> str:
        if url.lower().startswith(("http://", "https://")):
            return url
        return f"https://{url}"

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        normalized = domain.lower().strip(".")
        if normalized.startswith("www."):
            normalized = normalized[4:]
        return normalized

    @staticmethod
    def _safe_port(parsed_url) -> int | None:
        try:
            return parsed_url.port
        except ValueError:
            return None

    @staticmethod
    def _virustotal_url_id(url: str) -> str:
        encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
        return encoded.rstrip("=")

    @staticmethod
    def _normalize_stats(stats: Any) -> dict[str, int]:
        if not isinstance(stats, dict):
            return {}

        return {
            "malicious": int(stats.get("malicious", 0) or 0),
            "suspicious": int(stats.get("suspicious", 0) or 0),
            "harmless": int(stats.get("harmless", 0) or 0),
            "undetected": int(stats.get("undetected", 0) or 0),
        }

    @staticmethod
    def _has_excessive_subdomains(domain: str) -> bool:
        if not domain or IPV4_HOST_REGEX.match(domain):
            return False
        return len(domain.split(".")) > 4

    def _has_misleading_brand_domain(self, domain: str) -> bool:
        compact_domain = domain.replace("-", "").replace(".", "")
        for brand, real_domain in self.protected_brand_domains.items():
            if brand not in compact_domain:
                continue
            if domain == real_domain or domain.endswith(f".{real_domain}"):
                return False
            return True
        return False

    @staticmethod
    def _risk_score(flags: list[str], is_malicious: bool, is_suspicious: bool) -> float:
        if is_malicious:
            return 1.0

        weighted_flags = {
            "virustotal_suspicious": 0.35,
            "ip_address_url": 0.2,
            "misleading_brand_domain": 0.25,
            "excessive_subdomains": 0.15,
            "non_standard_port": 0.15,
            "url_encoded_characters": 0.12,
            "url_shortener": 0.18,
            "rate_limited": 0.05,
        }
        score = sum(weighted_flags.get(flag, 0.0) for flag in set(flags))
        if is_suspicious:
            score = max(score, 0.45)
        return round(min(score, 1.0), 4)

    @staticmethod
    def _overall_score(results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return round(max(float(result.get("risk_score", 0.0)) for result in results), 4)
