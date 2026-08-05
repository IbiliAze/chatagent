import pytest

from app.security.output_validator import OutputValidator
from app.security.pii_detector.pii_detector import PIIDetector

pytestmark = pytest.mark.unit


@pytest.fixture
def pii_detector() -> PIIDetector:
  return PIIDetector()


@pytest.fixture
def output_validator(pii_detector: PIIDetector) -> OutputValidator:
  return OutputValidator(pii_detector=pii_detector)


class TestInvalidOutput:
  @pytest.mark.parametrize(
    'text',
    [
      'the password is',
      'steps to manufacture drug',
    ],
  )
  def test_harmful_patterns(self, output_validator: OutputValidator, text: str) -> None:
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
    result = output_validator.validate(text)

    assert result.is_valid is False
    assert result.output is not None
    assert result.reason is not None
