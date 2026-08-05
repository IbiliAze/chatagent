"""Orchestrates the individual security checks into an input and output pipeline."""

from dataclasses import dataclass

from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]

from app.security.input_sanitiser import InputSanitiser
from app.security.language_detector import LanguageDetector
from app.security.output_validator import OutputValidator
from app.security.pii_detector.pii_detector import PIIDetector
from app.security.security_guard import SecurityGuard


@dataclass(frozen=True)
class InputCheckResult:
    """Result of running user input through the full input security pipeline."""

    is_allowed: bool
    cleaned_text: str
    security_notes: list[str]


@dataclass(frozen=True)
class OutputCheckResult:
    """Result of running LLM output through the full output security pipeline."""

    is_valid: bool
    output: str
    reason: str | None


class SecurityPipeline:
    """Full security pipeline that processes input and output."""

    def __init__(
        self,
        input_sanitiser: InputSanitiser,
        output_validator: OutputValidator,
        security_guard: SecurityGuard,
        pii_detector: PIIDetector,
        language_detector: LanguageDetector,
    ) -> None:
        self.input_sanitiser = input_sanitiser
        self.output_validator = output_validator
        self.security_guard = security_guard
        self.pii_detector = pii_detector
        self.language_detector = language_detector

    @traceable(name='security_check_input')
    def check_input(self, input: str) -> InputCheckResult:
        """Process the user input prompt through security checks."""
        notes: list[str] = []

        # 1. Detect other languages
        language_detection_result = self.language_detector.check(input)
        if not language_detection_result.allowed:
            return InputCheckResult(
                is_allowed=False,
                cleaned_text='',
                security_notes=[
                    language_detection_result.reason
                    or f'Rogue language detected: {language_detection_result.detected_language}'
                ],
            )

        # 2. Check for suspicious input
        suspense = self.input_sanitiser.is_suspicious(input)
        if suspense.is_suspicious:
            notes.append(suspense.reason) if suspense.reason is not None else None
            return InputCheckResult(
                is_allowed=False, security_notes=notes, cleaned_text=''
            )

        # 3. Sanitise the input
        cleaned = self.input_sanitiser.sanitise(input)

        # 4. Mask PII
        cleaned_masked = self.pii_detector.mask(cleaned)

        # 5. Go through security guard
        security_check_result = self.security_guard.security_check(cleaned_masked)
        if not security_check_result.safe:
            notes.append(security_check_result.reason)
            return InputCheckResult(
                is_allowed=False, security_notes=notes, cleaned_text=''
            )

        # 6. Security checked
        return InputCheckResult(
            is_allowed=True, security_notes=notes, cleaned_text=cleaned_masked
        )

    @traceable(name='security_check_output')
    def check_output(self, output: str) -> OutputCheckResult:
        """Process the LLM output response through security checks."""
        output_validation_result = self.output_validator.validate(output)

        # 1. Detect other languages
        language_detection_result = self.language_detector.check(output)
        if not language_detection_result.allowed:
            return OutputCheckResult(
                is_valid=False,
                output='',
                reason=language_detection_result.reason
                or f'Rogue language detected: {language_detection_result.detected_language}',
            )

        # 2. Validate output
        output_validation_result = self.output_validator.validate(output)
        if not output_validation_result.is_valid:
            return OutputCheckResult(
                is_valid=False,
                output=output_validation_result.output,
                reason=output_validation_result.reason,
            )

        # 3. Detect other languages
        language_detection_result = self.language_detector.check(output)
        if not language_detection_result.allowed:
            return OutputCheckResult(
                is_valid=False,
                output=output_validation_result.output,
                reason=language_detection_result.reason
                or f'Rogue language detected: {language_detection_result.detected_language}',
            )

        return OutputCheckResult(
            is_valid=True, output=output_validation_result.output, reason=None
        )
