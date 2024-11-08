// frontend/src/App.js

import React, { useState } from 'react';
import StockChart from './components/StockChart';

function App() {
    const [token, setToken] = useState(''); // Obtain token after login
    const [symbol, setSymbol] = useState('AAPL');

    return (
        <div className="App">
            <h1>AI-Powered Trading Dashboard</h1>
            <StockChart symbol={symbol} token={token} />
        </div>
    );
}

export default App;
