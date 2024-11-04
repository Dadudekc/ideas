import sys
import webbrowser
import json
from pathlib import Path
import numpy as np
import yfinance as yf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QTextEdit, QMessageBox, QGroupBox,
    QFormLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataFetcher:
    def __init__(self):
        pass

    def fetch_historical_data(self, symbol, period='1y', interval='1d'):
        data = yf.download(symbol, period=period, interval=interval)
        return data

class PremarketRoutinePlanner(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data_fetcher = DataFetcher()
        self.init_ml_model()
        self.load_routine()

    # Initialize the UI Components
    def init_ui(self):
        self.setWindowTitle("Premarket Routine Planner")
        self.setGeometry(100, 100, 600, 900)
        
        main_layout = QVBoxLayout()

        # Add sections for different steps
        self.add_step_sections(main_layout)

        # Add ML Prediction Section
        self.add_prediction_section(main_layout)

        # Add buttons for actions
        self.add_action_buttons(main_layout)

        self.setLayout(main_layout)

    # Initialize Machine Learning Model
    def init_ml_model(self):
        # Fetch historical data
        data = self.data_fetcher.fetch_historical_data('TSLA', period='1y', interval='1d')

        # Prepare features and target
        data['Price_Change'] = data['Close'] - data['Open']
        data['Market_Direction'] = data['Price_Change'].apply(
            lambda x: 'Bullish' if x > 0 else ('Bearish' if x < 0 else 'Neutral')
        )
        data['Econ_Reviewed'] = 1  # Assuming economic calendar is always reviewed
        data['Barchart_Reviewed'] = 1  # Assuming Barchart cheat sheet is always reviewed
        data['Support_Level'] = data['Low']
        data['Resistance_Level'] = data['High']
        data['Profitable'] = data['Price_Change'].apply(lambda x: 1 if x > 0 else 0)

        # Encode categorical data
        self.le_market_direction = LabelEncoder()
        data['Market_Direction_Encoded'] = self.le_market_direction.fit_transform(data['Market_Direction'])

        # Prepare features and target
        features = data[['Econ_Reviewed', 'Market_Direction_Encoded', 'Barchart_Reviewed', 'Support_Level', 'Resistance_Level']]
        target = data['Profitable']

        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Initialize and train the model
        self.model = LogisticRegression()
        self.model.fit(X_scaled, target)

    # Add Prediction Section
    def add_prediction_section(self, layout):
        prediction_group = QGroupBox("Machine Learning Prediction")
        prediction_layout = QVBoxLayout()

        self.predict_button = QPushButton("Predict Profitability")
        self.predict_button.setToolTip("Click to predict if the trading day will be profitable based on current inputs.")
        self.predict_button.clicked.connect(self.make_prediction)
        prediction_layout.addWidget(self.predict_button)

        self.prediction_result = QLabel("Prediction: N/A")
        self.prediction_result.setAlignment(Qt.AlignCenter)
        self.prediction_result.setStyleSheet("font-size: 16px; font-weight: bold;")
        prediction_layout.addWidget(self.prediction_result)

        prediction_group.setLayout(prediction_layout)
        layout.addWidget(prediction_group)

    # Make Prediction
    def make_prediction(self):
        try:
            # Collect features from UI inputs
            econ_reviewed = 1 if self.econ_checkbox.isChecked() else 0
            market_direction = self.market_direction_input.text().strip().capitalize()
            if market_direction not in self.le_market_direction.classes_:
                QMessageBox.warning(self, "Input Error", "Market Direction must be Bullish, Bearish, or Neutral.")
                return
            market_direction_encoded = self.le_market_direction.transform([market_direction])[0]
            barchart_reviewed = 1 if self.barchart_checkbox.isChecked() else 0

            support_level = float(self.support_level_input.text()) if self.support_level_input.text() else 0.0
            resistance_level = float(self.resistance_level_input.text()) if self.resistance_level_input.text() else 0.0

            # Prepare feature array
            features = np.array([[econ_reviewed, market_direction_encoded, barchart_reviewed, support_level, resistance_level]])
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_label = "Profitable" if prediction == 1 else "Not Profitable"
            self.prediction_result.setText(f"Prediction: {prediction_label}")

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure that support and resistance levels are valid numbers.")

    # Add all the sections to the main layout
    def add_step_sections(self, layout):
        self.add_economic_calendar_section(layout)
        self.add_market_health_section(layout)
        self.add_barchart_review_section(layout)
        self.add_price_levels_section(layout)
        self.add_sentiment_check_section(layout)
        self.add_profit_check_section(layout)

    # Add Economic Calendar Section
    def add_economic_calendar_section(self, layout):
        econ_group = QGroupBox("Step 1: Economic Calendar Check")
        econ_layout = QVBoxLayout()
        
        econ_button = self.create_button(
            "Open Economic Calendar",
            "Click to open the economic calendar in your web browser.",
            lambda: webbrowser.open("https://www.marketwatch.com/economy-politics/calendar")
        )
        econ_layout.addWidget(econ_button)
        
        self.econ_checkbox = self.create_checkbox("Economic Calendar Reviewed")
        econ_layout.addWidget(self.econ_checkbox)

        econ_group.setLayout(econ_layout)
        layout.addWidget(econ_group)

    # Add Market Health Section
    def add_market_health_section(self, layout):
        market_group = QGroupBox("Step 2: Market Health Check")
        market_layout = QVBoxLayout()

        # Market Health URLs
        buttons_info = [
            ("Open NASDAQ Futures", "https://www.investing.com/indices/nq-100-futures", "Click to view NASDAQ Futures on Investing.com."),
            ("Open TSLA on TradingView", "https://www.tradingview.com/symbols/TSLA/", "Click to view TSLA chart on TradingView.")
        ]
        for text, url, tooltip in buttons_info:
            market_layout.addWidget(self.create_button(text, tooltip, lambda u=url: webbrowser.open(u)))

        # Inputs
        form_layout = QFormLayout()
        self.market_direction_input = self.create_line_edit("Bullish, Bearish, or Neutral", "Enter the current market direction.")
        form_layout.addRow("Market Direction:", self.market_direction_input)
        
        self.market_notes = self.create_text_edit("Additional notes on market health", "Enter any additional notes regarding the market health.")
        form_layout.addRow("Market Health Notes:", self.market_notes)

        market_layout.addLayout(form_layout)
        market_group.setLayout(market_layout)
        layout.addWidget(market_group)

    # Add Barchart Cheat Sheet Review Section
    def add_barchart_review_section(self, layout):
        barchart_group = QGroupBox("Step 3: Barchart Cheat Sheet Review")
        barchart_layout = QVBoxLayout()

        barchart_button = self.create_button(
            "Open Barchart Cheat Sheet",
            "Click to open the Barchart Cheat Sheet in your web browser.",
            lambda: webbrowser.open("https://www.barchart.com/stocks/quotes/TSLA/cheat-sheet")
        )
        barchart_layout.addWidget(barchart_button)

        self.barchart_checkbox = self.create_checkbox("Barchart Cheat Sheet Reviewed")
        barchart_layout.addWidget(self.barchart_checkbox)

        barchart_group.setLayout(barchart_layout)
        layout.addWidget(barchart_group)

    # Add Price Levels Section
    def add_price_levels_section(self, layout):
        price_group = QGroupBox("Step 4: Chart Key Price Levels")
        price_layout = QFormLayout()

        self.support_level_input = self.create_line_edit("e.g., 150.00", "Enter the first support level.")
        price_layout.addRow("Support Level 1 ($):", self.support_level_input)

        self.resistance_level_input = self.create_line_edit("e.g., 155.00", "Enter the first resistance level.")
        price_layout.addRow("Resistance Level 1 ($):", self.resistance_level_input)

        self.price_notes = self.create_text_edit("Additional notes on price levels", "Enter any additional notes regarding price levels.")
        price_layout.addRow("Additional Price Level Notes:", self.price_notes)

        price_group.setLayout(price_layout)
        layout.addWidget(price_group)

    # Add Sentiment Check Section
    def add_sentiment_check_section(self, layout):
        sentiment_group = QGroupBox("Step 5: Sentiment Check")
        sentiment_layout = QVBoxLayout()

        # Sentiment URLs
        buttons_info = [
            ("Open Stocktwits", "https://www.stocktwits.com/", "Click to open Stocktwits for market sentiment."),
            ("Open TraderTV", "https://www.youtube.com/@TraderTVLive", "Click to open TraderTV Live on YouTube."),
            ("Open BrandonTrades Youtube", "https://www.youtube.com/@Brandontrades", "Click to open BrandonTrades YouTube channel.")
        ]
        for text, url, tooltip in buttons_info:
            sentiment_layout.addWidget(self.create_button(text, tooltip, lambda u=url: webbrowser.open(u)))

        # Sentiment Notes
        self.sentiment_notes = self.create_text_edit("Notes on sentiment alignment", "Enter notes on how sentiment aligns with your analysis.")
        sentiment_layout.addWidget(QLabel("Sentiment Notes:"))
        sentiment_layout.addWidget(self.sentiment_notes)

        sentiment_group.setLayout(sentiment_layout)
        layout.addWidget(sentiment_group)

    # Add Profit Check Section
    def add_profit_check_section(self, layout):
        profit_group = QGroupBox("Step 6: 10 a.m. Profit Check Reminder")
        profit_layout = QFormLayout()

        self.profit_target_input = self.create_line_edit("e.g., 500", "Enter your daily profit target in USD.")
        profit_layout.addRow("Daily Profit Target ($):", self.profit_target_input)

        self.max_loss_input = self.create_line_edit("e.g., 100", "Enter your maximum daily loss limit in USD.")
        profit_layout.addRow("Max Daily Loss Limit ($):", self.max_loss_input)

        profit_group.setLayout(profit_layout)
        layout.addWidget(profit_group)

    # Add Action Buttons Section
    def add_action_buttons(self, layout):
        button_layout = QHBoxLayout()

        actions_info = [
            ("Show Summary", self.show_summary),
            ("Save Routine", self.save_routine),
            ("Load Routine", self.load_routine),
            ("Reset", self.reset_fields)
        ]
        for text, handler in actions_info:
            button_layout.addWidget(self.create_button(text, "", handler))

        layout.addLayout(button_layout)

    # Utility function to create a button
    def create_button(self, text, tooltip, handler):
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(handler)
        return button

    # Utility function to create a checkbox
    def create_checkbox(self, text):
        checkbox = QCheckBox(text)
        checkbox.setToolTip(f"Check this box once you have reviewed the {text.lower()}.")
        return checkbox

    # Utility function to create a line edit
    def create_line_edit(self, placeholder, tooltip):
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setToolTip(tooltip)
        return line_edit

    # Utility function to create a text edit
    def create_text_edit(self, placeholder, tooltip):
        text_edit = QTextEdit()
        text_edit.setPlaceholderText(placeholder)
        text_edit.setToolTip(tooltip)
        return text_edit

    # Show summary of the routine
    def show_summary(self):
        try:
            profit_target = float(self.profit_target_input.text()) if self.profit_target_input.text() else 0
            max_loss = float(self.max_loss_input.text()) if self.max_loss_input.text() else 0
            support_level = float(self.support_level_input.text()) if self.support_level_input.text() else "N/A"
            resistance_level = float(self.resistance_level_input.text()) if self.resistance_level_input.text() else "N/A"
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for profit, loss, and price levels.")
            return

        summary = f"""
=== Summary of Today's Premarket Routine ===

1. Economic Calendar Reviewed: {'Yes' if self.econ_checkbox.isChecked() else 'No'}

2. Market Health Check:
   - Market Direction: {self.market_direction_input.text() or 'N/A'}
   - Market Health Notes: {self.market_notes.toPlainText() or 'N/A'}

3. Barchart Cheat Sheet Reviewed: {'Yes' if self.barchart_checkbox.isChecked() else 'No'}

4. Chart Key Price Levels:
   - Support Level 1: {support_level}
   - Resistance Level 1: {resistance_level}
   - Additional Price Level Notes: {self.price_notes.toPlainText() or 'N/A'}

5. Sentiment Check:
   - Sentiment Notes: {self.sentiment_notes.toPlainText() or 'N/A'}

6. 10 a.m. Profit Check Reminder:
   - Daily Profit Target: ${profit_target}
   - Max Daily Loss Limit: ${max_loss}

7. ML Prediction:
   - {self.prediction_result.text()}
"""
        QMessageBox.information(self, "Premarket Routine Summary", summary)

    # Save routine to a JSON file
    def save_routine(self):
        routine = {
            "econ_reviewed": self.econ_checkbox.isChecked(),
            "market_direction": self.market_direction_input.text(),
            "market_health_notes": self.market_notes.toPlainText(),
            "barchart_reviewed": self.barchart_checkbox.isChecked(),
            "support_level_1": self.support_level_input.text(),
            "resistance_level_1": self.resistance_level_input.text(),
            "price_notes": self.price_notes.toPlainText(),
            "sentiment_notes": self.sentiment_notes.toPlainText(),
            "profit_target": self.profit_target_input.text(),
            "max_loss": self.max_loss_input.text()
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Routine", "", "JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    json.dump(routine, file, indent=4)
                QMessageBox.information(self, "Success", "Routine saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save routine: {e}")

    # Load routine from a JSON file
    def load_routine(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Routine", "", "JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    routine = json.load(file)
                
                self.econ_checkbox.setChecked(routine.get("econ_reviewed", False))
                self.market_direction_input.setText(routine.get("market_direction", ""))
                self.market_notes.setPlainText(routine.get("market_health_notes", ""))
                self.barchart_checkbox.setChecked(routine.get("barchart_reviewed", False))
                self.support_level_input.setText(routine.get("support_level_1", ""))
                self.resistance_level_input.setText(routine.get("resistance_level_1", ""))
                self.price_notes.setPlainText(routine.get("price_notes", ""))
                self.sentiment_notes.setPlainText(routine.get("sentiment_notes", ""))
                self.profit_target_input.setText(routine.get("profit_target", ""))
                self.max_loss_input.setText(routine.get("max_loss", ""))
                
                QMessageBox.information(self, "Success", "Routine loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load routine: {e}")

    # Reset all input fields
    def reset_fields(self):
        confirm = QMessageBox.question(
            self, "Confirm Reset", "Are you sure you want to reset all fields?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.econ_checkbox.setChecked(False)
            self.market_direction_input.clear()
            self.market_notes.clear()
            self.barchart_checkbox.setChecked(False)
            self.support_level_input.clear()
            self.resistance_level_input.clear()
            self.price_notes.clear()
            self.sentiment_notes.clear()
            self.profit_target_input.clear()
            self.max_loss_input.clear()
            self.prediction_result.setText("Prediction: N/A")
            QMessageBox.information(self, "Reset", "All fields have been reset.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PremarketRoutinePlanner()
    window.show()
    sys.exit(app.exec_())
