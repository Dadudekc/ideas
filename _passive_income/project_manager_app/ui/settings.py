# Filename: settings.py
# Description: Settings tab for configuring user preferences in the Project Management App.

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox, QPushButton, QMessageBox

class Settings(QWidget):
    """
    Settings Class

    Provides a UI for user settings and preferences, such as notification settings,
    theme options, and other configurations for the Project Management App.
    """

    def __init__(self):
        """Initializes the settings tab with configurable options."""
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Sets up the layout and widgets for the Settings tab."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Notification Setting
        self.notifications_checkbox = QCheckBox("Enable Notifications")
        self.notifications_checkbox.setChecked(True)  # Default to enabled
        layout.addWidget(self.notifications_checkbox)

        # Theme Selection
        layout.addWidget(QLabel("Select Theme:"))
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark", "System Default"])
        layout.addWidget(self.theme_selector)

        # Save Button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        # Apply the layout to the widget
        self.setLayout(layout)

    def save_settings(self):
        """Saves the current settings and provides feedback."""
        notifications_enabled = self.notifications_checkbox.isChecked()
        selected_theme = self.theme_selector.currentText()

        # Here, you might implement code to save these settings persistently

        # Show confirmation
        QMessageBox.information(self, "Settings Saved", f"Settings have been saved.\n\n"
                                f"Notifications: {'Enabled' if notifications_enabled else 'Disabled'}\n"
                                f"Theme: {selected_theme}")

        # Print to console or log for debug purposes
        print(f"Settings saved: Notifications: {notifications_enabled}, Theme: {selected_theme}")
