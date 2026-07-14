import re


class PIIDetector:
    """Detect and mask personally indentifiable information."""

    PATTERNS = [
        ('EMAIL', re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')),
        (
            'PHONE',
            re.compile(r'(?<!\d)(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)'),
        ),
        ('SSN', re.compile(r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)')),
        ('CREDIT_CARD', re.compile(r'(?<!\d)(?:\d[ -]*?){13,16}(?!\d)')),
        ('IP_ADDRESS', re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
        ('IBAN', re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b')),
        (
            'DOB',
            re.compile(
                r'(?<!\d)(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}(?!\d)'
            ),
        ),
        ('PASSPORT', re.compile(r'\b[A-Z]{1,2}\d{6,9}\b')),
        (
            'US_ADDRESS',
            re.compile(
                r'\b\d{1,5}\s\w+(\s\w+)*\s(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                re.IGNORECASE,
            ),
        ),
    ]

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text."""
        found: dict[str, list[str]] = {}
        for pii_type, pattern in self.PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches

        return found

    def mask(self, text: str) -> str:
        """Mask PII in text."""
        masked = text

        for pii_type, pattern in self.PATTERNS:
            masked = pattern.sub(f'[{pii_type} REDACTED]', masked)

        return masked
