from PyQt5.QtCore import QThread, pyqtSignal

class BacktestThread(QThread):
    """
    Thread for running backtesting tasks in the background.
    Emits log updates via log_signal during execution.
    """
    log_signal = pyqtSignal(str)  # Signal for log updates

    def __init__(self, backtester):
        super().__init__()
        self.backtester = backtester
        self.backtester.log_callback = self.log_signal  # Connect log callback

    def run(self):
        """
        Executes the backtester's run method.
        QThread automatically emits 'finished' upon completion.
        """
        self.backtester.run()


class PaperTradeThread(QThread):
    """
    Thread for running paper trading tasks in the background.
    Emits log updates via log_signal during execution.
    """
    log_signal = pyqtSignal(str)  # Signal for log updates

    def __init__(self, trader):
        super().__init__()
        self.trader = trader
        self.trader.log_callback = self.log_signal  # Connect log callback

    def run(self):
        """
        Executes the trader's run method.
        QThread automatically emits 'finished' upon completion.
        """
        self.trader.run()
