from pathlib import Path
from loguru import logger
import sys

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "{message}"
)

logger.add(
    LOG_DIR / "insightiq.log",
    rotation="10 MB",
    retention=5
)