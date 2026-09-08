import html
import threading

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .data_manager import load_movies
from .llm import RAGService
from .search_engine import SearchEngine
from .security import (
    delete_openrouter_key,
    has_openrouter_key,
    save_openrouter_key,
)
from .workers import Worker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Webflyx RAG Studio")
        self.resize(1100, 760)

        self.pool = QThreadPool.globalInstance()
        self.engine_instance = None
        self.engine_lock = threading.Lock()
        self.active_workers = []

        self.tabs = QTabWidget()

        self.tabs.addTab(self._search_tab(), "Search")
        self.tabs.addTab(self._assistant_tab(), "AI Assistant")
        self.tabs.addTab(self._settings_tab(), "Settings")

        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage(
            "Ready — local search does not require an API key."
        )

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111318;
                color: #e8eaf0;
                font-size: 14px;
            }

            QTabWidget::pane {
                border: 1px solid #30343c;
                border-radius: 8px;
            }

            QLineEdit, QComboBox, QSpinBox, QTextBrowser {
                background: #1a1d23;
                border: 1px solid #363b46;
                border-radius: 7px;
                padding: 8px;
            }

            QPushButton {
                background: #325ee6;
                border: none;
                border-radius: 7px;
                padding: 9px 16px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #416df2;
            }

            QPushButton:disabled {
                background: #444852;
                color: #888c95;
            }

            QTabBar::tab {
                padding: 10px 18px;
            }

            QTabBar::tab:selected {
                background: #252932;
                border-radius: 6px;
            }
            """
        )

    def _engine(self) -> SearchEngine:
        with self.engine_lock:
            if self.engine_instance is None:
                movies = load_movies()
                self.engine_instance = SearchEngine(movies)

        return self.engine_instance

    def _run(self, function, callback) -> None:
        worker = Worker(function)
        self.active_workers.append(worker)

        worker.signals.result.connect(callback)
        worker.signals.error.connect(self._error)

        def cleanup():
            if worker in self.active_workers:
                self.active_workers.remove(worker)

            self.statusBar().showMessage("Ready")

        worker.signals.finished.connect(cleanup)

        self.statusBar().showMessage(
            "Working… first use may download models."
        )

        self.pool.start(worker)

    def _error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Webflyx RAG Studio",
            message,
        )

    def _search_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search movies by concept, plot, title, theme…"
        )

        self.result_limit = QSpinBox()
        self.result_limit.setRange(1, 30)
        self.result_limit.setValue(10)

        self.rerank_box = QCheckBox("Quality rerank")
        self.rerank_box.setChecked(True)

        search_button = QPushButton("Search")
        image_button = QPushButton("Search Image")

        controls.addWidget(self.search_input, 1)
        controls.addWidget(QLabel("Results"))
        controls.addWidget(self.result_limit)
        controls.addWidget(self.rerank_box)
        controls.addWidget(search_button)
        controls.addWidget(image_button)

        self.search_results = QTextBrowser()

        layout.addLayout(controls)
        layout.addWidget(self.search_results)

        search_button.clicked.connect(self._text_search)
        self.search_input.returnPressed.connect(self._text_search)
        image_button.clicked.connect(self._image_search)

        return widget

    def _text_search(self) -> None:
        query = self.search_input.text().strip()

        if not query:
            return

        limit = self.result_limit.value()
        rerank = self.rerank_box.isChecked()

        self._run(
            lambda: self._engine().text_search(
                query,
                limit=limit,
                rerank=rerank,
            ),
            self._render_search_results,
        )

    def _image_search(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
        )

        if not path:
            return

        limit = self.result_limit.value()

        self._run(
            lambda: self._engine().image_search(
                path,
                limit=limit,
            ),
            self._render_search_results,
        )

    def _render_search_results(self, results) -> None:
        if not results:
            self.search_results.setHtml("<p>No results.</p>")
            return

        output = []

        for number, result in enumerate(results, start=1):
            title = html.escape(result["title"])
            description = html.escape(
                result.get("description", "")[:700]
            )

            score = ""

            if "similarity" in result:
                score = (
                    f" · image similarity "
                    f"{result['similarity']:.3f}"
                )
            elif "rerank_score" in result:
                score = (
                    f" · rerank "
                    f"{result['rerank_score']:.3f}"
                )

            output.append(
                f"<h3>{number}. {title}"
                f"<small>{score}</small></h3>"
                f"<p>{description}</p>"
            )

        self.search_results.setHtml("\n".join(output))

    def _assistant_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QHBoxLayout()

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText(
            "Ask about movies in the database…"
        )

        self.assistant_mode = QComboBox()
        self.assistant_mode.addItems(
            ["Question", "Summary", "Citations"]
        )

        ask_button = QPushButton("Ask")

        controls.addWidget(self.question_input, 1)
        controls.addWidget(self.assistant_mode)
        controls.addWidget(ask_button)

        self.answer_output = QTextBrowser()

        layout.addLayout(controls)
        layout.addWidget(self.answer_output)

        ask_button.clicked.connect(self._ask)
        self.question_input.returnPressed.connect(self._ask)

        return widget

    def _ask(self) -> None:
        query = self.question_input.text().strip()

        if not query:
            return

        mode = self.assistant_mode.currentText()

        def task():
            results = self._engine().text_search(
                query,
                limit=8,
                rerank=True,
            )

            answer = RAGService().generate(
                query,
                results,
                mode,
            )

            return results, answer

        self._run(task, self._render_answer)

    def _render_answer(self, payload) -> None:
        results, answer = payload

        sources = "<br>".join(
            f"[{index}] {html.escape(result['title'])}"
            for index, result in enumerate(results, start=1)
        )

        self.answer_output.setHtml(
            f"<h2>Answer</h2>"
            f"<p>{html.escape(answer).replace(chr(10), '<br>')}</p>"
            f"<hr>"
            f"<h3>Retrieved sources</h3>"
            f"<p>{sources}</p>"
        )

    def _settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText(
            "Paste OpenRouter API key"
        )

        form.addRow("OpenRouter API key", self.key_input)

        buttons = QHBoxLayout()

        save_button = QPushButton("Save securely")
        clear_button = QPushButton("Remove key")

        buttons.addWidget(save_button)
        buttons.addWidget(clear_button)

        self.key_status = QLabel()

        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.key_status)

        security_note = QLabel(
            "The key is stored using the operating system's "
            "credential vault. It is never saved in the project "
            "directory or committed to GitHub."
        )
        security_note.setWordWrap(True)

        layout.addWidget(security_note)
        layout.addStretch()

        save_button.clicked.connect(self._save_key)
        clear_button.clicked.connect(self._clear_key)

        self._refresh_key_status()

        return widget

    def _refresh_key_status(self) -> None:
        try:
            configured = has_openrouter_key()
        except Exception as exc:
            self.key_status.setText(str(exc))
            return

        self.key_status.setText(
            "API key configured."
            if configured
            else "No API key configured. Local search still works."
        )

    def _save_key(self) -> None:
        try:
            save_openrouter_key(self.key_input.text())
        except Exception as exc:
            self._error(str(exc))
            return

        self.key_input.clear()
        self._refresh_key_status()

        QMessageBox.information(
            self,
            "Webflyx RAG Studio",
            "API key saved in secure operating-system storage.",
        )

    def _clear_key(self) -> None:
        try:
            delete_openrouter_key()
        except Exception as exc:
            self._error(str(exc))
            return

        self._refresh_key_status()
