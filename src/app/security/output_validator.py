import re
from typing import Optional

from app.security.pii_detector import PIIDetector


class OutputValidator:
    HARMFUL_PATTERNS = [
        re.compile(r'password is', re.IGNORECASE),
        re.compile(r'api\s?key is', re.IGNORECASE),
        re.compile(r'secret key is', re.IGNORECASE),
        re.compile(r'access token is', re.IGNORECASE),
        re.compile(r'private key is', re.IGNORECASE),
        re.compile(
            r'how to (make|build|synthesi[sz]e) (a bomb|explosives?)', re.IGNORECASE
        ),
        re.compile(
            r'how to (kill|murder|poison) (yourself|someone|a person)', re.IGNORECASE
        ),
        re.compile(r'instructions? (for|to|on) (self-harm|suicide)', re.IGNORECASE),
        re.compile(
            r'steps? to (manufacture|synthesi[sz]e) (a )?(drug|narcotic|weapon)',
            re.IGNORECASE,
        ),
    ]

    def __init__(self) -> None:
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[bool, str, Optional[str]]:
        """Validates output"""
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            cleaned = self.pii_detector.mask(output)
            return False, cleaned, f'PII detected and masked {list(pii_found.keys())}'

        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                return (
                    False,
                    '[CONTENT BLOCKED]',
                    'Potentially harmful content detected',
                )

        return True, output, None
