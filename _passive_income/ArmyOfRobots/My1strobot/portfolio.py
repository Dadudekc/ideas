# portfolio.py

from datetime import datetime, timezone

class Portfolio:
    """
    Simulates a trading portfolio by tracking balance, positions, and P&L.
    """
    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.positions = {}  # key: symbol, value: {'quantity': int, 'entry_price': float}
        self.trade_history = []  # List of executed trades

    def buy(self, symbol, price, quantity):
        total_cost = price * quantity
        if self.balance >= total_cost:
            self.balance -= total_cost
            if symbol in self.positions:
                total_quantity = self.positions[symbol]['quantity'] + quantity
                avg_price = ((self.positions[symbol]['entry_price'] * self.positions[symbol]['quantity']) + (price * quantity)) / total_quantity
                self.positions[symbol]['quantity'] = total_quantity
                self.positions[symbol]['entry_price'] = avg_price
            else:
                self.positions[symbol] = {'quantity': quantity, 'entry_price': price}
            self.trade_history.append({
                'type': 'BUY',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'time': datetime.now(timezone.utc)
            })
            return True
        else:
            return False

    def sell(self, symbol, price, quantity):
        if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
            total_revenue = price * quantity
            self.balance += total_revenue
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]
            self.trade_history.append({
                'type': 'SELL',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'time': datetime.now(timezone.utc)
            })
            return True
        else:
            return False

    def get_position(self, symbol):
        return self.positions.get(symbol, {'quantity': 0, 'entry_price': 0.0})

    def get_total_equity(self, current_prices):
        equity = self.balance
        for symbol, pos in self.positions.items():
            equity += pos['quantity'] * current_prices.get(symbol, 0.0)
        return equity
