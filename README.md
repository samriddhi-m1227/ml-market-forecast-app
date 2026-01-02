# 📈 Market Open Forecast

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://ml-market-forecast-app-tnlgyrsfku5rdujzn3l4jv.streamlit.app/)

An interactive **Streamlit application** that forecasts **next-day market movement** using an XGBoost model trained on historical market data and aggregated financial news sentiment. The predicted market movement is translated into an **implied next-day open price**, with additional watchlist impact analysis and model explainability.

---

## 🚀 Live Demo
👉 [Try the app here](https://ml-market-forecast-app-tnlgyrsfku5rdujzn3l4jv.streamlit.app/)


---

## 🧠 Project Background

This application is a **personal deployment extension** of my Salesforce Break Through Tech (Cornell Tech) ML fellowship project, where I built an end-to-end machine learning pipeline to model **market behavior using financial news sentiment and market indicators**.

🔗 [Original Salesforce Project Repository](https://github.com/samriddhi-m1227/BTT-Salesforce1a)

The original project focused on:
- Large-scale data collection and preprocessing
- Feature engineering from financial news sentiment
- Training and evaluating ML models (XGBoost)
- Comparing predictive performance against baselines

This repo takes that work **one step further** by:
- Packaging the trained model for inference
- Designing a user-facing Streamlit application
- Deploying the model as a live, interactive demo

---

## 🔍 What This App Does

### 1️⃣ Market Forecast
- Uses a trained **XGBoost regression model** to predict the **next-day log return** of the market.
- Converts the predicted return into an **estimated next-day open price** using:
  
### 2️⃣ Watchlist Impact (Beta-Adjusted)
- Translates the market forecast into **ticker-level estimates** using each stock’s recent **beta** relative to a market proxy.
- Helps answer:
> *“If the market moves as forecasted, how might my watchlist react?”*

### 3️⃣ Backtesting
- Evaluates the model on historical data
- Compares predictions against actual outcomes
- Includes baseline comparisons for context

### 4️⃣ Explainability
- Visualizes **feature importance** from the trained XGBoost model
- Helps interpret which inputs most influenced predictions

---

## 📊 Data Coverage
- **2017–2019**
- Historical market data + aggregated financial news sentiment

> This app is intentionally designed as a **historical ML demo** to showcase modeling, inference, and deployment — not live trading.

---

## 🛠️ Tech Stack

- **Python**
- **XGBoost**
- **Streamlit**
- **Scikit learn**
- **FinBERT**
- **pandas / NumPy**
- **yfinance**
- **joblib**
- **Matplotlib**

---

## 🧪 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

