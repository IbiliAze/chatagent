import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IsSuspiciousResult:
  is_suspicious: bool
  reason: str | None


class InputSanitiser:
  """Sanitise user input before processing."""

  INJECTION_PATTERNS = [
    re.compile(
      r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions', re.IGNORECASE
    ),
    re.compile(
      r'disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)',
      re.IGNORECASE,
    ),
    re.compile(
      r'forget\s+(all\s+)?(previous|prior|your)\s+(instructions|training)',
      re.IGNORECASE,
    ),
    re.compile(r'you\s+are\s+now\s+(a|an|in)\s', re.IGNORECASE),
    re.compile(r'act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s', re.IGNORECASE),
    re.compile(r'pretend\s+(you\s+are|to\s+be)\s', re.IGNORECASE),
    re.compile(r'new\s+instructions?\s*:', re.IGNORECASE),
    re.compile(r'system\s*prompt', re.IGNORECASE),
    re.compile(
      r'reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)', re.IGNORECASE
    ),
    re.compile(r'repeat\s+(the\s+)?(words|text)\s+above', re.IGNORECASE),
    re.compile(r'\bDAN\s+mode\b', re.IGNORECASE),
    re.compile(r'jailbreak', re.IGNORECASE),
    re.compile(r'</?(system|assistant|user)>', re.IGNORECASE),
    re.compile(r'\[/?(system|assistant|user)\]', re.IGNORECASE),
    re.compile(
      r'override\s+(your|the)\s+(guidelines|rules|instructions)', re.IGNORECASE
    ),
    re.compile(r'do\s+anything\s+now', re.IGNORECASE),
  ]

  def is_suspicious(self, text: str) -> IsSuspiciousResult:
    """Check if input contains suspicious patterns"""
    for pattern in self.INJECTION_PATTERNS:
      if pattern.search(text):
        return IsSuspiciousResult(
          is_suspicious=True,
          reason=f'Suspicious pattern detected: {pattern.pattern}',
        )
    return IsSuspiciousResult(is_suspicious=False, reason=None)

  def sanitise(self, text: str) -> str:
    """Remove potential dangerous content"""
    text = re.sub(r'[-]{3,}', '', text)
    text = re.sub(r'[=]{3,}', '', text)

    text = text.replace('{{', '{ {').replace('}}', '} }')

    return text.strip()
