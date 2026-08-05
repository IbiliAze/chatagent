"""Tests for OutputValidator's harmful-content and PII checks."""

import pytest

from app.security.output_validator import OutputValidator
from app.security.pii_detector.pii_detector import PIIDetector

pytestmark = pytest.mark.unit


@pytest.fixture
def pii_detector() -> PIIDetector:
    """Build a PIIDetector for the tests."""
    return PIIDetector()


@pytest.fixture
def output_validator(pii_detector: PIIDetector) -> OutputValidator:
    """Build an OutputValidator wired to the PIIDetector fixture."""
    return OutputValidator(pii_detector=pii_detector)


class TestInvalidOutput:
    """validate's rejection of harmful content and PII."""

    @pytest.mark.parametrize(
        'text',
        [
            'the password is',
            'steps to manufacture drug',
        ],
    )
    def test_harmful_patterns(
        self, output_validator: OutputValidator, text: str
    ) -> None:
        """Text matching a harmful pattern is blocked and replaced entirely."""
        result = output_validator.validate(text)

        assert result.is_valid is False
        assert result.output == '[CONTENT BLOCKED]'
        assert result.reason is not None

    @pytest.mark.parametrize(
        'text',
        [
            'the email is jon@doe.com',
        ],
    )
    def test_pii_detection_patterns(
        self, output_validator: OutputValidator, text: str
    ) -> None:
        """Text containing PII is invalidated and the PII is masked in the output."""
        result = output_validator.validate(text)

        assert result.is_valid is False
        assert result.output is not None
        assert result.reason is not None
