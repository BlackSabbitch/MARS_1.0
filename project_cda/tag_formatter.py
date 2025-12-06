import logging
from typing import Union


class TagFormatter(logging.Formatter):
    """
    Custom formatter that supports a `tag` attribute.
    If a tag is present, logs messages as:  [TAG] message
    Otherwise:                              [LEVEL] message
    """

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None)
        prefix = f"[{tag}]" if tag else f"[{record.levelname}]"
        record.msg = f"{prefix} {record.msg}"
        return super().format(record)

logger = logging.getLogger("graph_logger")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = TagFormatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

def log(msg: str, *, tag: str | None = None, level: Union[int, str] = logging.INFO):
    """
    EXAMPLES:
    log(f"Loading config from {config_file_path}", tag="FOUND", level=logging.DEBUG)    | [DEBUG] --> [FOUND]
    log(f"Config loaded: {config_file_path}", tag="LOADED") | [INFO] --> [LOADED]
    log(f"{purpose} is too large for pandas → skipped", tag="SKIPPED", level=logging.WARNING)   | [WARNING] --> [SKIPPED]

    EXAMPLES:
    log("Start", tag="INIT", level="DEBUG")   # Можно строкой!
    log("Done", tag="OK")                     # По умолчанию INFO
    """
    # Если передали строку ('debug', 'WARN' и т.д.), превращаем её в уровень
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.log(level, msg, extra={"tag": tag})

def set_log_level(level_name: str):
    """
    Sets the logging level safely using a string.
    Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
