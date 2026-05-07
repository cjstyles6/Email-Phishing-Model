"""Threat intelligence helpers for sender IP and domain reputation."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from email.parser import Parser
from email.policy import default
from email.utils import parseaddr
from typing import Any

import aiohttp
import whois


IPV4_REGEX = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
)
IPV6_REGEX = re.compile(
    r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])"
)


class ThreatIntelligence:
    """Run external and local sender threat-intelligence checks."""

    def __init__(self, abuseipdb_api_key: str | None = None) -> None:
        self.abuseipdb_api_key = abuseipdb_api_key or os.getenv("ABUSEIPDB_API_KEY")

    async def check_ip_reputation(self, ip_address: str) -> dict[str, Any]:
        """Check sender IP reputation with AbuseIPDB."""
        if not self.abuseipdb_api_key:
            return {"ip": ip_address, "error": "unavailable"}

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.get(
                    "https://api.abuseipdb.com/api/v2/check",
                    params={
                        "ipAddress": ip_address,
                        "maxAgeInDays": "90",
                        "verbose": "true",
                    },
                    headers={
                        "Key": self.abuseipdb_api_key,
                        "Accept": "application/json",
                    },
                ) as response:
                    if response.status >= 400:
                        return {"ip": ip_address, "error": "unavailable"}

                    payload = await response.json()
        except Exception:
            return {"ip": ip_address, "error": "unavailable"}

        data = payload.get("data", {})
        abuse_score = int(data.get("abuseConfidenceScore", 0) or 0)

        return {
            "ip": data.get("ipAddress") or ip_address,
            "abuse_score": abuse_score,
            "total_reports": int(data.get("totalReports", 0) or 0),
            "country": data.get("countryCode"),
            "isp": data.get("isp"),
            "usage_type": data.get("usageType"),
            "domain": data.get("domain"),
            "is_whitelisted": bool(data.get("isWhitelisted", False)),
            "is_malicious": abuse_score >= 25,
            "is_suspicious": abuse_score >= 10,
            "last_reported": data.get("lastReportedAt"),
        }

    def extract_sender_ip(self, raw_headers: str | None) -> str | None:
        """Extract the originating sender IP from the last Received header."""
        if not raw_headers:
            return None

        parsed_headers = Parser(policy=default).parsestr(raw_headers)
        received_headers = parsed_headers.get_all("Received", [])
        if not received_headers:
            return None

        closest_to_sender = received_headers[-1]
        return self._extract_ip_from_text(closest_to_sender)

    def check_domain_age(self, domain: str) -> dict[str, Any]:
        """Query WHOIS and score sender domain age risk."""
        try:
            whois_data = whois.whois(domain)
        except Exception:
            return {"domain": domain, "error": "no_whois_data"}

        creation_date = self._first_date(whois_data.creation_date)
        if creation_date is None:
            return {"domain": domain, "error": "no_whois_data"}

        expiration_date = self._first_date(whois_data.expiration_date)
        age_days = max((datetime.now(timezone.utc) - creation_date).days, 0)
        is_high_risk = age_days < 7
        is_suspicious = age_days < 30

        if is_high_risk:
            risk_reason = "Domain was registered less than 7 days ago."
        elif is_suspicious:
            risk_reason = "Domain was registered less than 30 days ago."
        else:
            risk_reason = "Domain age does not indicate elevated risk."

        return {
            "domain": domain,
            "creation_date": creation_date.date().isoformat(),
            "expiration_date": expiration_date.date().isoformat()
            if expiration_date
            else None,
            "age_days": age_days,
            "registrar": self._string_or_none(whois_data.registrar),
            "country": self._string_or_none(getattr(whois_data, "country", None)),
            "is_suspicious": is_suspicious,
            "is_high_risk": is_high_risk,
            "risk_reason": risk_reason,
        }

    def extract_sender_domain(self, from_header: str | None) -> str | None:
        """Extract the sender email domain from a From header value."""
        if not from_header:
            return None

        _, email_address = parseaddr(from_header)
        if "@" not in email_address:
            return None

        return email_address.rsplit("@", 1)[1].lower().strip(".> ")

    async def analyze(
        self,
        raw_headers: str | None = None,
        gmail_headers: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Run sender IP reputation and sender domain age analysis."""
        normalized_headers = self._normalize_headers(raw_headers, gmail_headers)
        sender_ip = self.extract_sender_ip(raw_headers) or self._extract_ip_from_headers(
            normalized_headers
        )
        sender_domain = self.extract_sender_domain(
            self._first_header(normalized_headers, "from")
        )

        ip_analysis = (
            await self.check_ip_reputation(sender_ip)
            if sender_ip
            else {"error": "no_sender_ip"}
        )
        domain_analysis = (
            self.check_domain_age(sender_domain)
            if sender_domain
            else {"error": "no_sender_domain"}
        )

        flags = self._build_flags(ip_analysis, domain_analysis)
        threat_intelligence_score = self._score(ip_analysis, domain_analysis)
        threat_level = self._threat_level(threat_intelligence_score)

        return {
            "ip_analysis": ip_analysis,
            "domain_analysis": domain_analysis,
            "threat_intelligence_score": threat_intelligence_score,
            "threat_level": threat_level,
            "flags": flags,
            "summary": self._summary(flags, threat_level),
        }

    def _normalize_headers(
        self,
        raw_headers: str | None,
        gmail_headers: list[Any] | None,
    ) -> dict[str, list[str]]:
        headers: dict[str, list[str]] = {}

        if raw_headers:
            parsed_headers = Parser(policy=default).parsestr(raw_headers)
            for name, value in parsed_headers.items():
                headers.setdefault(name.lower(), []).append(str(value).strip())

        for header in gmail_headers or []:
            if isinstance(header, dict):
                name = header.get("name")
                value = header.get("value")
            else:
                name = getattr(header, "name", None)
                value = getattr(header, "value", None)

            if name and value:
                headers.setdefault(str(name).lower(), []).append(str(value).strip())

        return headers

    @staticmethod
    def _first_header(headers: dict[str, list[str]], name: str) -> str | None:
        values = headers.get(name.lower(), [])
        return values[0] if values else None

    def _extract_ip_from_headers(self, headers: dict[str, list[str]]) -> str | None:
        received_headers = headers.get("received", [])
        if not received_headers:
            return None

        return self._extract_ip_from_text(received_headers[-1])

    @staticmethod
    def _extract_ip_from_text(text: str) -> str | None:
        ipv4_match = IPV4_REGEX.search(text)
        if ipv4_match:
            return ipv4_match.group(0)

        ipv6_match = IPV6_REGEX.search(text)
        if ipv6_match:
            return ipv6_match.group(0).strip("[]")

        return None

    @staticmethod
    def _first_date(value: Any) -> datetime | None:
        if isinstance(value, list):
            value = next((item for item in value if item), None)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return None

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if isinstance(value, list):
            value = next((item for item in value if item), None)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _build_flags(ip_analysis: dict[str, Any], domain_analysis: dict[str, Any]) -> list[str]:
        flags: list[str] = []

        if ip_analysis.get("is_malicious"):
            flags.append("sender_ip_malicious")
        elif ip_analysis.get("is_suspicious"):
            flags.append("sender_ip_suspicious")
        elif ip_analysis.get("error") == "unavailable":
            flags.append("ip_reputation_unavailable")

        if domain_analysis.get("is_high_risk"):
            flags.append("new_domain_high_risk")
        elif domain_analysis.get("is_suspicious"):
            flags.append("new_domain_suspicious")
        elif domain_analysis.get("error") == "no_whois_data":
            flags.append("domain_whois_unavailable")

        return flags

    @staticmethod
    def _score(ip_analysis: dict[str, Any], domain_analysis: dict[str, Any]) -> float:
        score = 0.0

        if ip_analysis.get("is_malicious"):
            score += 0.55
        elif ip_analysis.get("is_suspicious"):
            score += 0.3

        if domain_analysis.get("is_high_risk"):
            score += 0.45
        elif domain_analysis.get("is_suspicious"):
            score += 0.25

        return round(min(score, 1.0), 4)

    @staticmethod
    def _threat_level(score: float) -> str:
        if score < 0.3:
            return "safe"
        if score < 0.6:
            return "suspicious"
        return "danger"

    @staticmethod
    def _summary(flags: list[str], threat_level: str) -> str:
        if not flags:
            return "No sender IP or domain intelligence risk signals were detected."
        if threat_level == "danger":
            return (
                "Threat intelligence found high-risk sender infrastructure. "
                "Treat this email as dangerous until independently verified."
            )
        return (
            "Threat intelligence found sender infrastructure signals that warrant "
            "additional review before trusting the email."
        )
