import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from app.version import APP_VERSION


def test_main_window_constructs_and_exposes_core_tabs():
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    try:
        assert APP_VERSION in window.windowTitle()
        labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert labels == ["Search", "RAG Assistant", "Diagnostics", "Settings"]
        assert window.search_method.count() == 3
        assert "Recursive RAG" in [
            window.assistant_mode.itemText(index)
            for index in range(window.assistant_mode.count())
        ]
    finally:
        window.close()
        application.processEvents()
