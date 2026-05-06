"""
Structured JSON logging configuration.

Every log line is a JSON object with: timestamp, level, logger, message,
plus any extra fields passed via extra={"key": "value"}.

Why JSON logs?
  - Machine-parseable by any log aggregator (Loki, CloudWatch, Datadog)
  - request_id propagation makes tracing a query through the logs trivial
  - Structured fields enable PromQL-style alerting on log content

Example output:
  {"timestamp": "2026-05-06T09:12:34Z", "level": "INFO",
   "logger": "app.api.routes", "message": "query_complete",
   "query_id": "3f7a1c2d", "latency_ms": 3420}
"""
import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
