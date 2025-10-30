# 📊 Stock Market Analysis - Complete Data Engineering & Data Science Project

## 🎯 Project Overview

A complete stock analysis project, from data collection and building an ETL pipeline to business analysis and applying machine learning to predict stock prices.

### Key Components:

1. **Data Engineering**: ETL Pipeline from yfinance → MySQL  
2. **Data Science**: EDA, Preprocessing, Feature Engineering  
3. **Business Analysis**: Performance, trend, and volatility analysis  
4. **Machine Learning**: Comparing 7 models, selecting the best one  
5. **Insights**: Generating insights for business decisions

---

## 🛠 Tech Stack

### Data Engineering:
- **yfinance**: Collect stock data  
- **MySQL**: Database storage  
- **pandas**: Data manipulation  
- **mysql-connector-python**: Database connection

### Data Science & ML:
- **pandas, numpy**: Data analysis  
- **matplotlib, seaborn**: Visualization  
- **scikit-learn**: Machine learning  
- **XGBoost**: Gradient boosting  
- **scipy**: Statistical analysis

---

## 📋 Requirements

```bash
# requirements.txt
yfinance==0.2.31
mysql-connector-python==8.1.0
pandas==2.1.3
numpy==1.24.3
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.3.2
xgboost==2.0.1
scipy==1.11.4
```

### Installation

```bash
pip install -r requirements.txt
```

---

## 🗄 Database Setup

### 1. Install MySQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

**macOS:**
```bash
brew install mysql
brew services start mysql
```

**Windows:**
- Download MySQL Installer from [mysql.com](https://dev.mysql.com/downloads/installer/)

### 2. Create Database

```sql
mysql -u root -p

CREATE DATABASE stock_analysis;
USE stock_analysis;

-- Create new user (optional)
CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON stock_analysis.* TO 'stock_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Configure Database in Code

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # or 'stock_user'
    'password': 'your_password',
    'database': 'stock_analysis'
}
```

---

## 🚀 Running the Project

### Step 1: Data Engineering - ETL Pipeline

```bash
python etl_pipeline.py
```

**Output:**
- Creates stock_prices and stock_info tables  
- Loads 2 years of data for 5 tech stocks (AAPL, MSFT, GOOGL, AMZN, META)  
- Logs the Extract → Transform → Load process  

### Step 2: EDA & Preprocessing

```bash
python eda_preprocessing.py
```

**Output:**
- `stock_data_cleaned.csv`
- Visualizations: price_trends.png, volume_trends.png, correlation_matrix.png

### Step 3: Business Analysis

```bash
python business_analysis.py
```

**Output:**
- performance_analysis.png, trend_analysis.png, volatility_analysis.png
- business_insights.txt

### Step 4: Machine Learning

```bash
python ml_models.py
```

**Output:**
- model_comparison.csv, model_comparison.png
- predictions_*.png
- ml_insights.txt

---

## 📊 Project Structure

```
stock-analysis-project/
│
├── etl_pipeline.py
├── eda_preprocessing.py
├── business_analysis.py
├── ml_models.py
├── requirements.txt
├── README.md
│
├── output/
│   ├── stock_data_cleaned.csv
│   ├── model_comparison.csv
│   ├── business_insights.txt
│   ├── ml_insights.txt
│   └── visualizations/
│       ├── price_trends.png
│       ├── volume_trends.png
│       ├── model_comparison.png
│       └── predictions_*.png
│
└── database/
    └── stock_analysis.sql
```

---

## 📈 Key Results & Insights

**Business Insights:**
- Best Performer: Highest return stock  
- Volatility Analysis: Stable vs. high-risk stocks  
- Correlation: For diversification  

**ML Model Performance:**

| Model | Avg Test R² | Avg RMSE | Best For |
|--------|-------------|-----------|-----------|
| XGBoost | 0.95+ | Lowest | Overall performance |
| Random Forest | 0.93+ | Low | Stability & robustness |
| Gradient Boosting | 0.92+ | Low | Accuracy |
| SVR | 0.88+ | Medium | Non-linear patterns |
| Ridge/Lasso | 0.85+ | Medium | Interpretability |
| Linear Regression | 0.82+ | Higher | Baseline |

**Recommended Model:** XGBoost or Random Forest

---

## 🎓 Learning Outcomes

✅ ETL pipeline design  
✅ EDA & feature engineering  
✅ Financial metrics analysis  
✅ ML model comparison & evaluation  
✅ Insight generation for decision making  

---

## 🔧 Customization

Change stock list in `etl_pipeline.py`:
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
```

Change date range:
```python
START_DATE = '2020-01-01'
END_DATE = '2024-10-30'
```

Add new ML model in `ml_models.py`:
```python
from sklearn.neural_network import MLPRegressor
'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
```

---

## 🐛 Troubleshooting

**MySQL Connection Error:** Check password in `DB_CONFIG`  
**yfinance Error:** Verify ticker symbol  
**Memory Error:** Reduce stock count or date range  
**Missing Dependencies:**  
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Next Steps

- Real-time predictions  
- API deployment with Flask/FastAPI  
- Dashboard with Streamlit/Dash  
- Deep learning models (LSTM, GRU)  
- Portfolio optimization  
- Sentiment analysis  
- Backtesting strategies  

---

## 👨‍💻 Author

Data Engineering & Data Science Team

---

## 📄 License

MIT License - Free for educational use

---

## 🙏 Acknowledgments

- yfinance: Financial data API  
- scikit-learn: ML library  
- XGBoost: Gradient boosting framework  
- MySQL: Database management system

---

## 📞 Support

If you encounter issues:  
- Check the Troubleshooting section  
- Review the logs  
- Verify the database connection  

If you have any questions:
- Contact me at: huyquan1607@gmail.com

**Happy Analyzing! 📊🚀**

