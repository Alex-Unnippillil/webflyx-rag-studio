import logging
import sys
from logging.handlers import RotatingFileHandler

from PySide6.QtWidgets import QApplication

from .config import APP_NAME, LOG_DIR
from .ui import MainWindow


def configure_logging() -> None:
    handler = RotatingFileHandler(
        LOG_DIR / "webflyx.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
    )


def main() -> int:
    configure_logging()

    application = QApplication(sys.argv)
    application.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()

    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
