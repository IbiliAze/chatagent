import pytest

from security.pii_detector.pii_detector import PIIDetector

pytestmark = pytest.mark.unit


@pytest.fixture
def detector() -> PIIDetector:
  return PIIDetector()


class TestDetect:
  @pytest.mark.parametrize(
    ('pii_type', 'text'),
    [
      ('EMAIL', 'contact me at jane.doe@example.com please'),
      ('PHONE', 'call me at (555) 123-4567'),
      ('SSN', 'my ssn is 123-45-6789'),
      ('CREDIT_CARD', 'card number 4111 1111 1111 1111'),
      ('IP_ADDRESS', 'server lives at 192.168.1.1'),
      ('IBAN', 'wire it to GB29NWBK60161331926819'),
      ('DOB', 'born on 01/15/1990'),
      ('PASSPORT', 'passport number AB1234567'),
      ('ADDRESS', 'i live at 123 Main Street'),
    ],
  )
  def test_detects_each_pii_type(
    self, detector: PIIDetector, pii_type: str, text: str
  ) -> None:
    found = detector.detect(text)

    assert pii_type in found
    assert len(found[pii_type]) == 1

  def test_clean_text_returns_empty_dict(self, detector: PIIDetector) -> None:
    found = detector.detect('the weather is nice today')

    assert found == {}

  def test_detects_multiple_pii_types_in_one_string(
    self, detector: PIIDetector
  ) -> None:
    text = 'email jane.doe@example.com or call (555) 123-4567'

    found = detector.detect(text)

    assert 'EMAIL' in found
    assert 'PHONE' in found

  def test_detects_repeated_pii_of_same_type(self, detector: PIIDetector) -> None:
    text = 'reach jane@example.com or john@example.com'

    found = detector.detect(text)

    print(found)
    print(found)
    print(found)

    assert len(found['EMAIL']) == 2


class TestMask:
  def test_masks_email(self, detector: PIIDetector) -> None:
    masked = detector.mask('contact jane.doe@example.com now')

    assert 'jane.doe@example.com' not in masked
    assert '[EMAIL REDACTED]' in masked

  def test_masks_multiple_pii_types(self, detector: PIIDetector) -> None:
    text = 'email jane.doe@example.com or call (555) 123-4567'

    masked = detector.mask(text)

    assert '[EMAIL REDACTED]' in masked
    assert '[PHONE REDACTED]' in masked
    assert 'jane.doe@example.com' not in masked
    assert '(555) 123-4567' not in masked

  def test_clean_text_passes_through_unchanged(self, detector: PIIDetector) -> None:
    text = 'the weather is nice today'

    assert detector.mask(text) == text

  def test_masks_ssn_adjacent_to_credit_card_like_digits(
    self, detector: PIIDetector
  ) -> None:
    text = 'ssn 123-45-6789 card 4111111111111111'

    masked = detector.mask(text)

    assert '123-45-6789' not in masked
    assert '4111111111111111' not in masked
