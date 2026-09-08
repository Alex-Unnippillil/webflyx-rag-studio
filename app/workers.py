import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


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
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(value)
        finally:
            self.signals.finished.emit()
