import sys
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QTextEdit, QMessageBox

class PremarketRoutinePlanner(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Premarket Routine Planner")
        self.setGeometry(100, 100, 500, 700)
        
        layout = QVBoxLayout()

        # Step 1: Economic Calendar Check
        layout.addWidget(QLabel("Step 1: Economic Calendar Check"))
        econ_button = QPushButton("Open Economic Calendar")
        econ_button.clicked.connect(lambda: webbrowser.open("https://www.investing.com/economic-calendar/"))
        layout.addWidget(econ_button)
        self.econ_checkbox = QCheckBox("Economic Calendar Reviewed")
        layout.addWidget(self.econ_checkbox)

        # Step 2: Market Health Check
        layout.addWidget(QLabel("\nStep 2: Market Health Check"))
        nasdaq_button = QPushButton("Open NASDAQ Futures")
        nasdaq_button.clicked.connect(lambda: webbrowser.open("https://www.cnbc.com/nasdaq-futures/"))
        layout.addWidget(nasdaq_button)
        qqq_button = QPushButton("Open QQQ on TradingView")
        qqq_button.clicked.connect(lambda: webbrowser.open("https://www.tradingview.com/symbols/NASDAQ-QQQ/"))
        layout.addWidget(qqq_button)
        layout.addWidget(QLabel("Market Direction (Bullish, Bearish, Neutral)"))
        self.market_direction_input = QLineEdit()
        layout.addWidget(self.market_direction_input)
        layout.addWidget(QLabel("Additional Notes on Market Health"))
        self.market_notes = QTextEdit()
        layout.addWidget(self.market_notes)

        # Step 3: Barchart Cheat Sheet Review
        layout.addWidget(QLabel("\nStep 3: Barchart Cheat Sheet Review"))
        barchart_button = QPushButton("Open Barchart Cheat Sheet")
        barchart_button.clicked.connect(lambda: webbrowser.open("https://www.barchart.com/"))
        layout.addWidget(barchart_button)
        self.barchart_checkbox = QCheckBox("Barchart Cheat Sheet Reviewed")
        layout.addWidget(self.barchart_checkbox)

        # Step 4: Chart Key Price Levels
        layout.addWidget(QLabel("\nStep 4: Chart Key Price Levels"))
        layout.addWidget(QLabel("Support Level 1"))
        self.support_level_input = QLineEdit()
        layout.addWidget(self.support_level_input)
        layout.addWidget(QLabel("Resistance Level 1"))
        self.resistance_level_input = QLineEdit()
        layout.addWidget(self.resistance_level_input)
        layout.addWidget(QLabel("Additional Price Level Notes"))
        self.price_notes = QTextEdit()
        layout.addWidget(self.price_notes)

        # Step 5: Sentiment Check
        layout.addWidget(QLabel("\nStep 5: Sentiment Check"))
        stocktwits_button = QPushButton("Open Stocktwits")
        stocktwits_button.clicked.connect(lambda: webbrowser.open("https://www.stocktwits.com/"))
        layout.addWidget(stocktwits_button)
        tradertv_button = QPushButton("Open TraderTV")
        tradertv_button.clicked.connect(lambda: webbrowser.open("https://tradertv.live/"))
        layout.addWidget(tradertv_button)
        brandontrades_button = QPushButton("Open BrandonTrades Twitter")
        brandontrades_button.clicked.connect(lambda: webbrowser.open("https://twitter.com/brandontrades"))
        layout.addWidget(brandontrades_button)
        layout.addWidget(QLabel("Notes on Sentiment Alignment"))
        self.sentiment_notes = QTextEdit()
        layout.addWidget(self.sentiment_notes)

        # Step 6: 10 a.m. Profit Check Reminder
        layout.addWidget(QLabel("\nStep 6: 10 a.m. Profit Check Reminder"))
        layout.addWidget(QLabel("Set Daily Profit Target ($)"))
        self.profit_target_input = QLineEdit()
        layout.addWidget(self.profit_target_input)
        layout.addWidget(QLabel("Set Max Daily Loss Limit ($)"))
        self.max_loss_input = QLineEdit()
        layout.addWidget(self.max_loss_input)

        # Summary and Save Button
        summary_button = QPushButton("Show Summary")
        summary_button.clicked.connect(self.show_summary)
        layout.addWidget(summary_button)

        self.setLayout(layout)

    def show_summary(self):
        summary = f"""
=== Summary of Today's Premarket Routine ===
1. Economic Calendar Reviewed: {'Yes' if self.econ_checkbox.isChecked() else 'No'}
2. Market Direction: {self.market_direction_input.text()}
   Market Health Notes: {self.market_notes.toPlainText()}
3. Barchart Cheat Sheet Reviewed: {'Yes' if self.barchart_checkbox.isChecked() else 'No'}
4. Key Price Levels:
   - Support Level 1: {self.support_level_input.text()}
   - Resistance Level 1: {self.resistance_level_input.text()}
   Additional Price Level Notes: {self.price_notes.toPlainText()}
5. Sentiment Notes: {self.sentiment_notes.toPlainText()}
6. Profit Target: ${self.profit_target_input.text()}, Max Loss Limit: ${self.max_loss_input.text()}
"""
        # Display summary in a message box
        QMessageBox.information(self, "Premarket Routine Summary", summary)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PremarketRoutinePlanner()
    window.show()
    sys.exit(app.exec_())
