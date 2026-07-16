from presidio_analyzer import Pattern, PatternRecognizer


class PatternRecogniser:
    PATTERNS = [
        PatternRecognizer(
            supported_entity='EMAIL',
            patterns=[
                Pattern(
                    name='email',
                    regex=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='PHONE',
            patterns=[
                Pattern(
                    name='phone',
                    regex=r'(?<!\d)(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='SSN',
            patterns=[
                Pattern(name='ssn', regex=r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)', score=0.85)
            ],
        ),
        PatternRecognizer(
            supported_entity='CREDIT_CARD',
            patterns=[
                Pattern(
                    name='credit_card',
                    regex=r'(?<!\d)(?:\d[ -]*?){13,16}(?!\d)',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='IP_ADDRESS',
            patterns=[
                Pattern(
                    name='ip_address',
                    regex=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='IBAN',
            patterns=[
                Pattern(
                    name='iban',
                    regex=r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='DOB',
            patterns=[
                Pattern(
                    name='dob',
                    regex=r'(?<!\d)(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}(?!\d)',
                    score=0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity='PASSPORT',
            patterns=[
                Pattern(name='passport', regex=r'\b[A-Z]{1,2}\d{6,9}\b', score=0.85)
            ],
        ),
        PatternRecognizer(
            supported_entity='ADDRESS',
            patterns=[
                Pattern(
                    name='address',
                    regex=r'\b\d{1,5}\s\w+(\s\w+)*\s(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                    score=0.85,
                )
            ],
        ),
    ]
