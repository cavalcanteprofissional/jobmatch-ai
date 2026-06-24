"""
Sistema centralizado de logging para o JobMatch AI.

Fornece:
    - Logging em arquivo rotativo (10MB, 5 backups)
    - Console output colorido
    - Loggers hierárquicos por módulo
    - Níveis: DEBUG, INFO, WARNING, ERROR

Uso:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Mensagem")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s | %(name)-45s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = "jobmatch.log",
    console: bool = True,
) -> logging.Logger:
    """
    Configura e retorna um logger com handlers de arquivo e console.

    Args:
        name: Nome do logger (geralmente __name__).
        level: Nível mínimo de log.
        log_file: Nome do arquivo dentro de LOG_DIR.
        console: Se True, adiciona handler de console.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Handler de arquivo rotativo (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Handler de console
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retorna logger já configurado ou configura com defaults."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
