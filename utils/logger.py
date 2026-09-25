"""구조화 트레이딩 로거.

loguru가 설치되어 있으면 그대로 사용하고, 없으면 동일한 API(`bind`, `success`,
`exception`)를 제공하는 표준 logging 기반 shim으로 자동 대체된다.
체결/주문 원장은 사람이 읽는 로그와 분리해 JSONL로 별도 적재한다(대회 리포트 근거자료).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - 설치 환경에 따라 분기
    from loguru import logger as _loguru_logger

    HAS_LOGURU = True
except Exception:  # pragma: no cover
    _loguru_logger = None
    HAS_LOGURU = False


# --------------------------------------------------------------------------------------
# loguru 호환 shim
# --------------------------------------------------------------------------------------
class _BoundLogger:
    """loguru의 `logger.bind(...)` 스타일 컨텍스트를 흉내내는 경량 래퍼."""

    _LEVELS = {
        "TRACE": 5,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "SUCCESS": 25,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, base: logging.Logger, context: Optional[Dict[str, Any]] = None):
        self._base = base
        self._context = dict(context or {})

    def bind(self, **kwargs: Any) -> "_BoundLogger":
        merged = dict(self._context)
        merged.update(kwargs)
        return _BoundLogger(self._base, merged)

    def _format(self, message: str, args: tuple, kwargs: dict) -> str:
        if args or kwargs:
            try:
                message = message.format(*args, **kwargs)
            except (IndexError, KeyError, ValueError):
                pass
        if self._context:
            ctx = " ".join(f"{k}={v}" for k, v in self._context.items())
            return f"{message} | {ctx}"
        return message

    def _log(self, level: str, message: str, args: tuple, kwargs: dict, exc_info: bool = False) -> None:
        self._base.log(self._LEVELS[level], self._format(message, args, kwargs), exc_info=exc_info)

    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("TRACE", message, args, kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("DEBUG", message, args, kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("INFO", message, args, kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("SUCCESS", message, args, kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("WARNING", message, args, kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("ERROR", message, args, kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("CRITICAL", message, args, kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("ERROR", message, args, kwargs, exc_info=True)


def _build_shim_logger() -> _BoundLogger:
    logging.addLevelName(25, "SUCCESS")
    logging.addLevelName(5, "TRACE")
    base = logging.getLogger("kis")
    base.setLevel(logging.INFO)
    base.propagate = False
    return _BoundLogger(base)


logger = _loguru_logger if HAS_LOGURU else _build_shim_logger()

_CONFIGURED = False
_LOCK = threading.Lock()


def enable_utf8_console() -> None:
    """Windows 콘솔(cp949)에서 한글/기호가 깨지거나 예외가 나지 않도록 UTF-8로 전환."""
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", "") or "").lower()
            if enc.replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    rotation_mb: int = 50,
    retention: int = 10,
    colorize: bool = True,
) -> None:
    """로그 sink를 구성한다. 여러 번 호출해도 1회만 적용된다."""
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        enable_utf8_console()
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)

        if HAS_LOGURU:  # pragma: no cover - loguru 설치 환경에서만 실행
            _loguru_logger.remove()
            fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
            _loguru_logger.add(sys.stderr, level=level, format=fmt, colorize=colorize, enqueue=True)
            _loguru_logger.add(
                path / "trader_{time:YYYY-MM-DD}.log",
                level=level,
                rotation=f"{rotation_mb} MB",
                retention=retention,
                encoding="utf-8",
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )
            _loguru_logger.add(
                path / "error_{time:YYYY-MM-DD}.log",
                level="ERROR",
                rotation=f"{rotation_mb} MB",
                retention=retention,
                encoding="utf-8",
                enqueue=True,
            )
        else:
            base = logging.getLogger("kis")
            base.handlers.clear()
            base.setLevel(_BoundLogger._LEVELS.get(level.upper(), logging.INFO))
            fmt = logging.Formatter(
                "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(fmt)
            base.addHandler(stream)

            from logging.handlers import RotatingFileHandler

            fileh = RotatingFileHandler(
                path / "trader.log",
                maxBytes=rotation_mb * 1024 * 1024,
                backupCount=retention,
                encoding="utf-8",
            )
            fileh.setFormatter(fmt)
            base.addHandler(fileh)

            errh = RotatingFileHandler(
                path / "error.log",
                maxBytes=rotation_mb * 1024 * 1024,
                backupCount=retention,
                encoding="utf-8",
            )
            errh.setLevel(logging.ERROR)
            errh.setFormatter(fmt)
            base.addHandler(errh)

        _CONFIGURED = True


class JsonLedger:
    """주문/체결/리스크 결정 원장 (JSON Lines).

    스레드 안전하며 append-only. 백테스트와 실매매가 동일한 포맷을 사용하므로
    사후 분석 스크립트를 한 벌만 유지하면 된다.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "event": event_type,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def read_all(self) -> list[Dict[str, Any]]:
        if not self._path.exists():
            return []
        rows: list[Dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows


def get_logger(**context: Any):
    """컨텍스트가 바인딩된 로거를 반환한다."""
    return logger.bind(**context) if context else logger
