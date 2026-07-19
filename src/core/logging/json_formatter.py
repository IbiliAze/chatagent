import json
import logging
from datetime import UTC, datetime
from typing import Any, cast


class JSONFormatter(logging.Formatter):
  """Format logs as JSON for log aggregation."""

  def format(self, record: logging.LogRecord) -> str:
    log_object: dict[str, Any] = {
      'timestamp': datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
      'level': record.levelname,
      'message': record.getMessage(),
      'module': record.module,
      'function': record.funcName,
    }

    extra_data = getattr(record, 'extra_data', None)
    if isinstance(extra_data, dict):
      log_object.update(cast(dict[str, Any], extra_data))

    return json.dumps(log_object)
