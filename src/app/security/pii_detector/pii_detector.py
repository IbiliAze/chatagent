from dataclasses import dataclass
from typing import cast

from presidio_analyzer import EntityRecognizer
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities.engine.recognizer_result import (
    RecognizerResult as AnonymizerRecognizerResult,
)

from app.security.pii_detector.entity_recogniser import GLiNERRecognizer
from app.security.pii_detector.pattern_recogniser import PatternRecogniser

type EntityType = str
type Matches = list[str]
type PIIMatches = dict[EntityType, Matches]


@dataclass(frozen=True)
class ScanResult:
    pii_found: PIIMatches
    cleaned: str


class PIIDetector:
    """Detect and mask PII using regex recognizers combined with GLiNER."""

    def __init__(
        self,
        gliner_labels: tuple[str, ...] = ('person', 'address', 'organization'),
    ) -> None:
        self.recognizers: list[EntityRecognizer] = [
            *PatternRecogniser.PATTERNS,
            GLiNERRecognizer(gliner_labels),
        ]
        self.anonymizer = AnonymizerEngine()

    def _analyze(self, text: str) -> list[RecognizerResult]:
        """Run every recognizer and merge their results into one span list."""
        results: list[RecognizerResult] = []
        for recognizer in self.recognizers:
            results.extend(
                recognizer.analyze(
                    text,
                    entities=[],
                    nlp_artifacts=None,  # pyright: ignore[reportArgumentType]
                )
            )

        return self._resolve_overlaps(results)

    @staticmethod
    def _resolve_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
        """Keep the highest-confidence span when regex and GLiNER overlap."""
        by_confidence = sorted(
            results, key=lambda r: (r.score, r.end - r.start), reverse=True
        )

        kept: list[RecognizerResult] = []
        for result in by_confidence:
            overlaps = any(result.start < k.end and k.start < result.end for k in kept)
            if not overlaps:
                kept.append(result)

        return sorted(kept, key=lambda r: r.start)

    def detect(self, text: str) -> PIIMatches:
        """Detect PII in text, grouped by entity type."""
        return self._group(text, self._analyze(text))

    def mask(self, text: str) -> str:
        """Mask PII in text."""
        return self._redact(text, self._analyze(text))

    def scan(self, text: str) -> ScanResult:
        """Detect and mask PII in a single analysis pass."""
        results = self._analyze(text)
        return ScanResult(
            pii_found=self._group(text, results), cleaned=self._redact(text, results)
        )

    @staticmethod
    def _group(text: str, results: list[RecognizerResult]) -> PIIMatches:
        found: PIIMatches = {}
        for result in results:
            found.setdefault(result.entity_type, []).append(
                text[result.start : result.end]
            )

        return found

    def _redact(self, text: str, results: list[RecognizerResult]) -> str:
        operators = {
            result.entity_type: OperatorConfig(
                'replace', {'new_value': f'[{result.entity_type} REDACTED]'}
            )
            for result in results
        }

        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=cast(list[AnonymizerRecognizerResult], results),
            operators=operators,
        ).text
