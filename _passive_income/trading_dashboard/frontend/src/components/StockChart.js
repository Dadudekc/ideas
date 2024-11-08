// frontend/src/components/StockChart.js

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

const StockChart = ({ symbol, token }) => {
    const [chartData, setChartData] = useState({});

    useEffect(() => {
        const fetchStockData = async () => {
            try {
                const response = await axios.get(`http://localhost:8000/stocks/${symbol}`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                const data = response.data.data;
                const dates = data.map(entry => entry.Date);
                const closes = data.map(entry => entry.Close);

                setChartData({
                    labels: dates,
                    datasets: [
                        {
                            label: `${symbol} Close Price`,
                            data: closes,
                            fill: false,
                            backgroundColor: 'rgba(75,192,192,0.4)',
                            borderColor: 'rgba(75,192,192,1)',
                        },
                    ],
                });
            } catch (error) {
                console.error("Error fetching stock data:", error);
            }
        };

        fetchStockData();
    }, [symbol, token]);

    return (
        <div>
            <h2>{symbol} Stock Price</h2>
            <Line data={chartData} />
        </div>
    );
};

export default StockChart;