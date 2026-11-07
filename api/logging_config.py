import json
import logging
import os
import socket
import time
from datetime import datetime, timezone

_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class LogstashTCPHandler(logging.Handler):
    """Envoie chaque log en JSON (une ligne) vers Logstash. Ne fait jamais planter l'API."""

    def __init__(self, host, port, retry_delay=30.0):
        super().__init__()
        self.host, self.port, self.retry_delay = host, port, retry_delay
        self.sock = None
        self._last_failure = 0.0

    def emit(self, record):
        if self.sock is None and time.time() - self._last_failure < self.retry_delay:
            return
        try:
            payload = {
                "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "service": "ml-churn-api",
            }
            for key, value in vars(record).items():      # champs passés via extra={...}
                if key not in _STANDARD:
                    payload[key] = value
            if record.exc_info:
                payload["exception"] = logging.Formatter().formatException(record.exc_info)
            if self.sock is None:
                self.sock = socket.create_connection((self.host, self.port), timeout=2)
            self.sock.sendall((json.dumps(payload, default=str) + "\n").encode("utf-8"))
        except Exception:
            self._last_failure = time.time()
            if self.sock:
                self.sock.close()
            self.sock = None


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("ml-churn-api")
    host = os.getenv("LOGSTASH_HOST")
    if host and not any(isinstance(h, LogstashTCPHandler) for h in logger.handlers):
        logger.addHandler(LogstashTCPHandler(host, int(os.getenv("LOGSTASH_PORT", "5000"))))
        logger.info("Envoi des logs vers Logstash activé (%s)", host)
    return logger