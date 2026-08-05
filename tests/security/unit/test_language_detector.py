import pytest

from app.security.language_detector import LanguageCheckResult, LanguageDetector


@pytest.fixture
def language_detector() -> LanguageDetector:
  return LanguageDetector()


class TestLanguageDetector:
  @pytest.mark.parametrize(
    'text',
    [
      'Hi how are you?',
      'My english is perfect',
      'You will spot no mistakes',
      'Because I grew up in the UK',
      'Do better',
    ],
  )
  def test_english_passes(self, language_detector: LanguageDetector, text: str) -> None:
    result = language_detector.check(text)

    assert result is not None
    assert isinstance(result, LanguageCheckResult)
    assert result.allowed is True

  @pytest.mark.parametrize(
    'text',
    [
      'Hi howw are you?',
      'I cant talk veri well',
      'but im trying',
      'forgiv me',
      'for my bad englis',
    ],
  )
  def test_slight_errors_in_english_pass(
    self, language_detector: LanguageDetector, text: str
  ) -> None:
    result = language_detector.check(text)

    assert result is not None
    assert isinstance(result, LanguageCheckResult)
    assert result.allowed is True

  @pytest.mark.parametrize(
    'text',
    [
      'Hi my name is Ibi',
      'Eduardo is a good programmer',
      'But not as good as Vasili',
      'Almost as good as Mo',
    ],
  )
  def test_non_english_names_pass(
    self, language_detector: LanguageDetector, text: str
  ) -> None:
    result = language_detector.check(text)

    assert result is not None
    assert isinstance(result, LanguageCheckResult)
    assert result.allowed is True

  @pytest.mark.parametrize(
    'text',
    [
      'Como estas?',
      'Comment allez-vous?',
      'Nasılsınız?',
      'Başım ağrayır',
    ],
  )
  def test_non_english_fails(
    self, language_detector: LanguageDetector, text: str
  ) -> None:
    result = language_detector.check(text)

    assert result is not None
    assert isinstance(result, LanguageCheckResult)
    assert result.allowed is False
    assert result.reason is not None
