import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


def _safe_emit(signal, *args) -> None:
    """Emit a Qt signal unless its QObject has already been destroyed."""
    try:
        signal.emit(*args)
    except RuntimeError:
        # The application may be shutting down while a background task finishes.
        # Qt deletes signal sources during teardown; emitting after that must not
        # crash the worker thread.
        pass


class Worker(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            value = self.function()
        except Exception as exc:
            traceback.print_exc()
            _safe_emit(self.signals.error, str(exc))
        else:
            _safe_emit(self.signals.result, value)
        finally:
            _safe_emit(self.signals.finished)
