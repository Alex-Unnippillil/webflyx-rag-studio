import html
import threading

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .cache_manager import clear_application_cache
from .config import OPENROUTER_MODEL
from .data_manager import load_movies
from .diagnostics import diagnostics_report
from .llm import RAGService
from .search_engine import SearchEngine
from .security import (
    delete_openrouter_key,
    has_openrouter_key,
    save_openrouter_key,
)
from .version import APP_VERSION
from .workers import Worker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"Webflyx RAG Studio {APP_VERSION}")
        self.resize(1180, 800)
        self.setMinimumSize(900, 650)

        self.pool = QThreadPool.globalInstance()
        self.engine_instance = None
        self.engine_lock = threading.Lock()
        self.active_workers = []
        self.closing = False

        self.last_assistant_query = ""
        self.last_assistant_results: list[dict] = []

        container = QWidget()
        layout = QVBoxLayout(container)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._search_tab(), "Search")
        self.tabs.addTab(self._assistant_tab(), "RAG Assistant")
        self.tabs.addTab(self._diagnostics_tab(), "Diagnostics")
        self.tabs.addTab(self._settings_tab(), "Settings")

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setMaximumHeight(4)
        self.busy_bar.hide()

        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.busy_bar)
        self.setCentralWidget(container)

        self.statusBar().showMessage(
            "Ready — local retrieval does not require an API key."
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

            QGroupBox {
                border: 1px solid #30343c;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }

            QLineEdit, QComboBox, QSpinBox, QTextBrowser {
                background: #1a1d23;
                border: 1px solid #363b46;
                border-radius: 7px;
                padding: 8px;
                selection-background-color: #325ee6;
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

            QProgressBar {
                border: 0;
                background: #252932;
            }

            QProgressBar::chunk {
                background: #416df2;
            }
            """
        )

    def _engine(self) -> SearchEngine:
        with self.engine_lock:
            if self.engine_instance is None:
                self.engine_instance = SearchEngine(load_movies())
        return self.engine_instance

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self.busy_bar.setVisible(busy)
        if message:
            self.statusBar().showMessage(message)
        elif not busy:
            self.statusBar().showMessage("Ready")

    def _run(self, function, callback, message: str = "Working…") -> None:
        if self.closing:
            return

        worker = Worker(function)
        self.active_workers.append(worker)
        worker.signals.result.connect(callback)
        worker.signals.error.connect(self._error)

        def cleanup() -> None:
            if worker in self.active_workers:
                self.active_workers.remove(worker)
            self._set_busy(bool(self.active_workers))
            self._refresh_diagnostics()

        worker.signals.finished.connect(cleanup)
        self._set_busy(True, message)
        self.pool.start(worker)

    def _error(self, message: str) -> None:
        if self.closing:
            return
        QMessageBox.critical(self, "Webflyx RAG Studio", message)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h1>Movie Search</h1>"))
        subtitle = QLabel(
            "Search with BM25, semantic embeddings, RRF hybrid retrieval, "
            "optional reranking, or an image."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        controls = QGridLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Try: family adventure with a friendly bear"
        )

        self.search_method = QComboBox()
        self.search_method.addItems(
            ["Hybrid (RRF)", "Keyword (BM25)", "Semantic"]
        )

        self.result_limit = QSpinBox()
        self.result_limit.setRange(1, 30)
        self.result_limit.setValue(10)

        self.rerank_box = QCheckBox("Cross-encoder rerank")
        self.rerank_box.setChecked(False)
        self.rerank_box.setToolTip(
            "Improves ranking quality but downloads/loads an additional model the first time."
        )

        search_button = QPushButton("Search")
        image_button = QPushButton("Search by image")

        controls.addWidget(self.search_input, 0, 0, 1, 4)
        controls.addWidget(QLabel("Method"), 1, 0)
        controls.addWidget(self.search_method, 1, 1)
        controls.addWidget(QLabel("Results"), 1, 2)
        controls.addWidget(self.result_limit, 1, 3)
        controls.addWidget(self.rerank_box, 2, 0, 1, 2)
        controls.addWidget(search_button, 2, 2)
        controls.addWidget(image_button, 2, 3)

        self.search_results = QTextBrowser()
        self.search_results.setPlaceholderText("Search results appear here.")

        layout.addLayout(controls)
        layout.addWidget(self.search_results, 1)

        search_button.clicked.connect(self._text_search)
        self.search_input.returnPressed.connect(self._text_search)
        image_button.clicked.connect(self._image_search)
        self.search_method.currentTextChanged.connect(self._sync_rerank_control)

        return widget

    def _sync_rerank_control(self) -> None:
        self.rerank_box.setEnabled(
            self.search_method.currentText() == "Hybrid (RRF)"
        )

    def _text_search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return

        limit = self.result_limit.value()
        method = self.search_method.currentText()
        rerank = self.rerank_box.isChecked()

        def task():
            engine = self._engine()
            if method == "Keyword (BM25)":
                return engine.keyword_search(query, limit)
            if method == "Semantic":
                return engine.semantic_search(query, limit)
            return engine.hybrid_search(query, limit, rerank=rerank)

        self._run(
            task,
            self._render_search_results,
            "Searching… first semantic/model use can take a little longer.",
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

        self._run(
            lambda: self._engine().image_search(
                path,
                limit=self.result_limit.value(),
            ),
            self._render_search_results,
            "Loading CLIP / searching by image…",
        )

    def _render_search_results(self, results: list[dict]) -> None:
        if not results:
            self.search_results.setHtml("<p>No matching movies were found.</p>")
            return

        output = []
        warning = results[0].get("warning")
        if warning:
            output.append(
                "<p><b>Note:</b> " + html.escape(str(warning)) + "</p><hr>"
            )

        for number, result in enumerate(results, start=1):
            title = html.escape(str(result.get("title", "Untitled")))
            description = html.escape(str(result.get("description", ""))[:900])
            score_parts = []

            if "similarity" in result:
                score_parts.append(f"image {result['similarity']:.3f}")
            if "rerank_score" in result:
                score_parts.append(f"rerank {result['rerank_score']:.3f}")
            elif "rrf_score" in result:
                score_parts.append(f"RRF {result['rrf_score']:.4f}")
            elif "semantic_score" in result:
                score_parts.append(f"semantic {result['semantic_score']:.3f}")
            elif "bm25_score" in result:
                score_parts.append(f"BM25 {result['bm25_score']:.3f}")

            score_text = " · ".join(score_parts)
            score_html = f" <small>· {score_text}</small>" if score_text else ""

            output.append(
                f"<h3>{number}. {title}{score_html}</h3>"
                f"<p>{description}</p>"
            )

        self.search_results.setHtml("\n".join(output))

    # ------------------------------------------------------------------
    # RAG assistant
    # ------------------------------------------------------------------

    def _assistant_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h1>Grounded RAG Assistant</h1>"))
        subtitle = QLabel(
            "Retrieve movie documents first, optionally rerank them, then generate "
            "a grounded answer with OpenRouter."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        controls = QGridLayout()

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText(
            "Ask a question, request recommendations, compare movies…"
        )

        self.assistant_mode = QComboBox()
        self.assistant_mode.addItems(
            [*RAGService.MODES, "Recursive RAG"]
        )

        self.query_enhancement = QComboBox()
        self.query_enhancement.addItems(
            ["None", "Spell", "Rewrite", "Expand"]
        )

        self.assistant_limit = QSpinBox()
        self.assistant_limit.setRange(3, 20)
        self.assistant_limit.setValue(8)

        self.assistant_rerank = QCheckBox("Rerank context")
        self.assistant_rerank.setChecked(False)

        ask_button = QPushButton("Run RAG")
        evaluate_button = QPushButton("Evaluate 0–3")

        controls.addWidget(self.question_input, 0, 0, 1, 4)
        controls.addWidget(QLabel("Answer mode"), 1, 0)
        controls.addWidget(self.assistant_mode, 1, 1)
        controls.addWidget(QLabel("Query enhancement"), 1, 2)
        controls.addWidget(self.query_enhancement, 1, 3)
        controls.addWidget(QLabel("Context docs"), 2, 0)
        controls.addWidget(self.assistant_limit, 2, 1)
        controls.addWidget(self.assistant_rerank, 2, 2)
        controls.addWidget(ask_button, 2, 3)
        controls.addWidget(evaluate_button, 3, 3)

        self.answer_output = QTextBrowser()
        self.answer_output.setPlaceholderText(
            "The grounded answer and retrieved sources appear here."
        )

        layout.addLayout(controls)
        layout.addWidget(self.answer_output, 1)

        ask_button.clicked.connect(self._ask)
        self.question_input.returnPressed.connect(self._ask)
        evaluate_button.clicked.connect(self._evaluate_last)

        return widget

    def _ask(self) -> None:
        original_query = self.question_input.text().strip()
        if not original_query:
            return

        mode = self.assistant_mode.currentText()
        enhancement = self.query_enhancement.currentText().lower()
        limit = self.assistant_limit.value()
        rerank = self.assistant_rerank.isChecked()

        def task():
            service = RAGService()
            search_query = original_query
            queries = [original_query]

            if enhancement != "none":
                search_query = service.rewrite_query(original_query, enhancement)
                if search_query != original_query:
                    queries.append(search_query)

            engine = self._engine()

            if mode == "Recursive RAG":
                payload = service.recursive_answer(
                    original_query,
                    lambda query, result_limit: engine.hybrid_search(
                        query,
                        result_limit,
                        rerank=rerank,
                    ),
                    limit=limit,
                    max_rounds=3,
                )
                payload["queries"] = queries + [
                    query
                    for query in payload["queries"]
                    if query not in queries
                ]
                payload["mode"] = mode
                return payload

            results = engine.hybrid_search(
                search_query,
                limit=limit,
                rerank=rerank,
            )
            answer = service.generate(original_query, results, mode)
            return {
                "answer": answer,
                "results": results,
                "queries": queries,
                "mode": mode,
            }

        self._run(
            task,
            self._render_answer,
            "Running retrieval and grounded generation…",
        )

    def _render_answer(self, payload: dict) -> None:
        results = payload.get("results", [])
        answer = str(payload.get("answer", ""))
        queries = payload.get("queries", [])
        mode = html.escape(str(payload.get("mode", "RAG")))

        self.last_assistant_query = self.question_input.text().strip()
        self.last_assistant_results = results

        source_html = "<br>".join(
            f"[{index}] {html.escape(str(result.get('title', 'Untitled')))}"
            for index, result in enumerate(results, start=1)
        )
        query_html = " → ".join(html.escape(str(query)) for query in queries)
        answer_html = html.escape(answer).replace("\n", "<br>")

        self.answer_output.setHtml(
            f"<h2>{mode}</h2>"
            f"<p>{answer_html}</p>"
            f"<hr><h3>Retrieval path</h3><p>{query_html}</p>"
            f"<h3>Retrieved sources</h3><p>{source_html}</p>"
        )

    def _evaluate_last(self) -> None:
        if not self.last_assistant_results or not self.last_assistant_query:
            QMessageBox.information(
                self,
                "Nothing to evaluate",
                "Run a RAG request first.",
            )
            return

        query = self.last_assistant_query
        results = list(self.last_assistant_results)

        self._run(
            lambda: RAGService().judge_results(query, results),
            self._render_judgement,
            "Evaluating retrieved results…",
        )

    def _render_judgement(self, scores: list[int]) -> None:
        rows = []
        for index, (result, score) in enumerate(
            zip(self.last_assistant_results, scores, strict=True),
            start=1,
        ):
            rows.append(
                f"{index}. {html.escape(str(result.get('title', 'Untitled')))}: "
                f"<b>{score}/3</b>"
            )

        self.answer_output.append(
            "<hr><h3>LLM relevance evaluation</h3><p>"
            + "<br>".join(rows)
            + "</p>"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _diagnostics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h1>Diagnostics & Model Management</h1>"))
        note = QLabel(
            "Models are lazy-loaded. Preparing them here avoids the first-search delay. "
            "Do not close the application while a model is being prepared."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.diagnostics_output = QTextBrowser()
        layout.addWidget(self.diagnostics_output, 1)

        group = QGroupBox("Model/cache actions")
        buttons = QGridLayout(group)

        refresh_button = QPushButton("Refresh diagnostics")
        semantic_button = QPushButton("Prepare semantic search")
        rerank_button = QPushButton("Prepare reranker")
        image_button = QPushButton("Prepare image search")
        clear_button = QPushButton("Clear model/index cache")

        buttons.addWidget(refresh_button, 0, 0)
        buttons.addWidget(semantic_button, 0, 1)
        buttons.addWidget(rerank_button, 1, 0)
        buttons.addWidget(image_button, 1, 1)
        buttons.addWidget(clear_button, 2, 0, 1, 2)

        layout.addWidget(group)

        refresh_button.clicked.connect(self._refresh_diagnostics)
        semantic_button.clicked.connect(
            lambda: self._prepare_models(semantic=True)
        )
        rerank_button.clicked.connect(
            lambda: self._prepare_models(semantic=False, reranker=True)
        )
        image_button.clicked.connect(
            lambda: self._prepare_models(semantic=False, multimodal=True)
        )
        clear_button.clicked.connect(self._clear_cache)

        self._refresh_diagnostics()
        return widget

    def _refresh_diagnostics(self) -> None:
        movie_count = self.engine_instance.size if self.engine_instance else None
        self.diagnostics_output.setPlainText(
            diagnostics_report(movie_count=movie_count)
        )

    def _prepare_models(
        self,
        *,
        semantic: bool,
        reranker: bool = False,
        multimodal: bool = False,
    ) -> None:
        self._run(
            lambda: self._engine().warm_up(
                semantic=semantic,
                reranker=reranker,
                multimodal=multimodal,
            ),
            lambda _value: QMessageBox.information(
                self,
                "Model ready",
                "The requested model/index is ready.",
            ),
            "Preparing model/index… this can take several minutes on first use.",
        )

    def _clear_cache(self) -> None:
        if self.active_workers:
            QMessageBox.information(
                self,
                "Task running",
                "Wait for the current task to finish before clearing the cache.",
            )
            return

        choice = QMessageBox.question(
            self,
            "Clear cache",
            "Delete downloaded models and generated embedding caches? "
            "They will be downloaded/rebuilt when needed.",
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        try:
            clear_application_cache()
            self.engine_instance = None
            self._refresh_diagnostics()
        except Exception as exc:
            self._error(str(exc))
            return

        QMessageBox.information(
            self,
            "Cache cleared",
            "The Webflyx model/index cache was cleared.",
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<h1>Settings</h1>"))

        credentials = QGroupBox("OpenRouter")
        form = QFormLayout(credentials)

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("Paste OpenRouter API key")

        self.show_key_box = QCheckBox("Show key while typing")
        self.key_status = QLabel()

        form.addRow("API key", self.key_input)
        form.addRow("", self.show_key_box)
        form.addRow("Model", QLabel(OPENROUTER_MODEL))
        form.addRow("Status", self.key_status)

        buttons = QHBoxLayout()
        save_button = QPushButton("Save securely")
        test_button = QPushButton("Test connection")
        clear_button = QPushButton("Remove key")
        buttons.addWidget(save_button)
        buttons.addWidget(test_button)
        buttons.addWidget(clear_button)

        security_note = QLabel(
            "The saved key is stored through the operating system credential vault. "
            "It is not written to the repository, ordinary config files, or logs."
        )
        security_note.setWordWrap(True)

        layout.addWidget(credentials)
        layout.addLayout(buttons)
        layout.addWidget(security_note)
        layout.addWidget(QLabel(f"Webflyx RAG Studio {APP_VERSION}"))
        layout.addStretch()

        save_button.clicked.connect(self._save_key)
        test_button.clicked.connect(self._test_key)
        clear_button.clicked.connect(self._clear_key)
        self.show_key_box.toggled.connect(self._toggle_key_visibility)
        self._refresh_key_status()

        return widget

    def _toggle_key_visibility(self, visible: bool) -> None:
        self.key_input.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )

    def _refresh_key_status(self) -> None:
        try:
            configured = has_openrouter_key()
        except Exception as exc:
            self.key_status.setText(str(exc))
            return

        self.key_status.setText(
            "Configured"
            if configured
            else "Not configured — local search still works"
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
            "Key saved",
            "The API key was saved in secure operating-system storage.",
        )

    def _test_key(self) -> None:
        entered = self.key_input.text().strip() or None
        self._run(
            lambda: RAGService(api_key=entered).test_connection(),
            lambda response: QMessageBox.information(
                self,
                "OpenRouter connection",
                response,
            ),
            "Testing OpenRouter connection…",
        )

    def _clear_key(self) -> None:
        try:
            delete_openrouter_key()
        except Exception as exc:
            self._error(str(exc))
            return
        self._refresh_key_status()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self.active_workers:
            QMessageBox.information(
                self,
                "Background task still running",
                "A search, model download, or AI request is still running. "
                "Wait for it to finish before closing Webflyx. This prevents "
                "model-loader and Qt worker shutdown errors.",
            )
            event.ignore()
            return

        self.closing = True
        event.accept()
