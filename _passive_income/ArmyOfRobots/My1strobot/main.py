# main.py

from PyQt5.QtWidgets import QApplication
import sys
from gui import TradingBotGUI

def main():
    app = QApplication(sys.argv)
    gui = TradingBotGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
