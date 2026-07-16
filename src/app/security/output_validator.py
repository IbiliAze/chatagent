import re
from dataclasses import dataclass

from app.security.pii_detector.pii_detector import PIIDetector


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    output: str
    reason: str | None = None


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

    def validate(self, output: str) -> ValidationResult:
        """Validates output"""
        result = self.pii_detector.scan(output)
        if result.pii_found:
            return ValidationResult(
                is_valid=False,
                output=result.cleaned,
                reason=f'PII detected and masked: {list(result.pii_found.keys())}',
            )

        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                return ValidationResult(
                    is_valid=False,
                    output='[CONTENT BLOCKED]',
                    reason='Potentially harmful content detected',
                )

        return ValidationResult(
            is_valid=True,
            output=output,
        )
