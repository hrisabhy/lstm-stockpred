# Stock Prediction App with LSTM Model

This project uses a Long Short-Term Memory (LSTM) model to predict stock prices based on historical data.

## Overview

The LSTM model is trained on historical stock data to capture patterns and trends by observing 100 days & 200 days
moving average.

## How to Run the Project

Follow these steps to set up and run the Stock Prediction App:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd stock-prediction-app
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
```bash
On Windows,
venv\Scripts\activate

On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Stock Prediction App
```bash
python streamlit app.py
```