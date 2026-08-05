"""GLiNER-backed Presidio recognizer for free-text PII entities."""

from gliner import GLiNER  # pyright: ignore[reportMissingTypeStubs]
from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult


class GLiNERRecognizer(EntityRecognizer):
    """Presidio recognizer that delegates entity detection to a GLiNER model."""

    def __init__(self, labels: tuple[str, ...]):
        super().__init__(supported_entities=[label.upper() for label in labels])
        self.model = GLiNER.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            'urchade/gliner_medium-v2.1'
        )
        self.labels = list(labels)

    def load(self) -> None:
        """No-op: the model is already loaded in __init__."""

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """Run the GLiNER model and translate its predictions into recognizer results."""
        predictions = self.model.predict_entities(  # pyright: ignore[reportUnknownMemberType]
            text, self.labels, threshold=0.5
        )
        return [
            RecognizerResult(
                entity_type=p['label'].upper(),
                start=p['start'],
                end=p['end'],
                score=p['score'],
            )
            for p in predictions
        ]
