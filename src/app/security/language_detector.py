from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder

# A short candidate list beats the full set of spoken languages: every extra
# language is one more look-alike a brief English message can lose to. Languages
# off the list cost nothing here — Vietnamese, Japanese and Swahili all score
# 0.00-0.07 for English, which is the only figure this gate reads.
CANDIDATE_LANGUAGES = (
  Language.ENGLISH,
  Language.SPANISH,
  Language.FRENCH,
  Language.GERMAN,
  Language.PORTUGUESE,
  Language.ITALIAN,
  Language.DUTCH,
  Language.TURKISH,
  Language.AZERBAIJANI,
  Language.RUSSIAN,
  Language.ARABIC,
  Language.CHINESE,
  Language.HINDI,
  Language.URDU,
  Language.POLISH,
  Language.ROMANIAN,
)

# Fewer letters than this and there is nothing to identify, so '400', 'INV-2291'
# and 'Ibi here' pass instead of being guessed at.
MIN_LETTERS = 8

# English scores below this only for text that really isn't English. Measured on
# short samples: weakest true English 0.17 ('Hi Ibi here'), strongest non-English
# 0.15 (romanised Russian). Tuned to admit English at the price of occasionally
# admitting a foreign phrase, because turning away a customer writing English is
# the more expensive mistake.
MIN_ENGLISH_CONFIDENCE = 0.15


@dataclass(frozen=True)
class LanguageCheckResult:
  allowed: bool
  detected_language: str | None
  reason: str | None = None
  english_confidence: float | None = None


class LanguageDetector:
  """Allows English input and blocks other languages.

  Judges by how plausible English is, not by which language wins. On a short
  message the winner is often Afrikaans or Dutch even when English is a close
  second, so 'is the top guess English' rejects ordinary English sentences.
  """

  def __init__(
    self,
    min_letters: int = MIN_LETTERS,
    min_english_confidence: float = MIN_ENGLISH_CONFIDENCE,
  ) -> None:
    self.min_letters = min_letters
    self.min_english_confidence = min_english_confidence
    # No with_low_accuracy_mode(): it trades away exactly the short-text
    # accuracy this gate lives on, and read 'Hi, my name is ibi' as Afrikaans.
    self._detector = LanguageDetectorBuilder.from_languages(
      *CANDIDATE_LANGUAGES
    ).build()

  def check(self, text: str) -> LanguageCheckResult:
    if not text.strip():
      return LanguageCheckResult(
        allowed=False,
        detected_language=None,
        reason='Input is empty.',
      )

    if sum(character.isalpha() for character in text) < self.min_letters:
      return LanguageCheckResult(
        allowed=True,
        detected_language=None,
        reason='Too short to identify a language.',
      )

    # One pass gives both the English score and the runner-up to report.
    confidences = self._detector.compute_language_confidence_values(text)
    english = next(
      (c.value for c in confidences if c.language == Language.ENGLISH), 0.0
    )

    if english >= self.min_english_confidence:
      return LanguageCheckResult(
        allowed=True,
        detected_language=Language.ENGLISH.name,
        english_confidence=english,
      )

    return LanguageCheckResult(
      allowed=False,
      detected_language=confidences[0].language.name if confidences else None,
      reason='Only English input is supported.',
      english_confidence=english,
    )
