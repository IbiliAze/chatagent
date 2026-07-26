from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder


@dataclass(frozen=True)
class LanguageCheckResult:
  allowed: bool
  detected_language: str | None
  reason: str | None = None


class LanguageDetector:
  """Allows English input and blocks other languages."""

  def __init__(self) -> None:
    self._detector = (
      LanguageDetectorBuilder.from_all_spoken_languages()
      .with_low_accuracy_mode()
      .build()
    )

  def check(self, text: str) -> LanguageCheckResult:
    if not text.strip():
      return LanguageCheckResult(
        allowed=False,
        detected_language=None,
        reason='Input is empty.',
      )

    language = self._detector.detect_language_of(text)

    if language is None:
      return LanguageCheckResult(
        allowed=False,
        detected_language=None,
        reason='Could not reliably detect the input language.',
      )

    if language != Language.ENGLISH:
      return LanguageCheckResult(
        allowed=False,
        detected_language=language.name,
        reason='Only English input is supported.',
      )

    return LanguageCheckResult(
      allowed=True,
      detected_language='ENGLISH',
    )
