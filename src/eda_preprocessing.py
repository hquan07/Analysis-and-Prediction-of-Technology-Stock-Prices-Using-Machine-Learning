import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class StockEDA:
    def __init__(self, db_config):
        """Initialize EDA with database configuration"""
        self.db_config = db_config
        self.df_prices = None
        self.df_info = None
        self.df_clean = None
        script_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(script_path)
        self.project_root = os.path.dirname(src_dir)
        self.output_dir = os.path.join(self.project_root, 'output')
        self.eda_output_dir = os.path.join(self.output_dir, 'EDA')

    def load_data_from_mysql(self):
        """Load data from MySQL database"""
        try:
            connection = mysql.connector.connect(**self.db_config)

            # Load stock prices
            query_prices = """
                           SELECT ticker, date, open, high, low, close, volume, adj_close
                           FROM stock_prices
                           ORDER BY ticker, date \
                           """
            self.df_prices = pd.read_sql(query_prices, connection)

            # Load stock info
            query_info = """
                         SELECT ticker, \
                                company_name, \
                                sector, \
                                industry, \
                                market_cap,
                                pe_ratio, \
                                dividend_yield
                         FROM stock_info \
                         """
            self.df_info = pd.read_sql(query_info, connection)

            connection.close()

            print(f"✓ Loaded {len(self.df_prices)} price records")
            print(f"✓ Loaded {len(self.df_info)} company records")

        except Exception as e:
            print(f"Error loading data: {e}")

    def initial_exploration(self):
        """Perform initial data exploration"""
        print("\n" + "=" * 80)
        print("INITIAL DATA EXPLORATION")
        print("=" * 80)

        # Stock Prices Dataset
        print("\n📊 STOCK PRICES DATASET")
        print("-" * 80)
        print(f"Shape: {self.df_prices.shape}")
        print(f"\nData Types:\n{self.df_prices.dtypes}")
        print(f"\nFirst 5 rows:\n{self.df_prices.head()}")
        print(f"\nBasic Statistics:\n{self.df_prices.describe()}")

        # Stock Info Dataset
        print("\n📊 STOCK INFO DATASET")
        print("-" * 80)
        print(f"Shape: {self.df_info.shape}")
        print(f"\nData Types:\n{self.df_info.dtypes}")
        print(f"\nCompany Information:\n{self.df_info}")

        # Missing values
        print("\n🔍 MISSING VALUES ANALYSIS")
        print("-" * 80)
        print("Stock Prices:")
        print(self.df_prices.isnull().sum())
        print("\nStock Info:")
        print(self.df_info.isnull().sum())

    def data_quality_check(self):
        """Check data quality issues"""
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECK")
        print("=" * 80)

        # Check for duplicates
        duplicates = self.df_prices.duplicated(subset=['ticker', 'date']).sum()
        print(f"\n📌 Duplicate records: {duplicates}")

        # Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        for col in numeric_cols:
            negative_count = (self.df_prices[col] < 0).sum()
            if negative_count > 0:
                print(f"⚠️  {col}: {negative_count} negative values found")

        # Check for outliers using IQR method
        print("\n📊 OUTLIER DETECTION (IQR Method)")
        print("-" * 80)
        for col in ['close', 'volume']:
            Q1 = self.df_prices[col].quantile(0.25)
            Q3 = self.df_prices[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df_prices[col] < (Q1 - 1.5 * IQR)) |
                        (self.df_prices[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outliers} outliers detected")

        # Date range check
        print("\n📅 DATE RANGE")
        print("-" * 80)
        for ticker in self.df_prices['ticker'].unique():
            ticker_data = self.df_prices[self.df_prices['ticker'] == ticker]
            print(f"{ticker}: {ticker_data['date'].min()} to {ticker_data['date'].max()} "
                  f"({len(ticker_data)} records)")

    def visualize_data(self):
        """Create visualizations for EDA"""
        print("\n" + "=" * 80)
        print("DATA VISUALIZATION")
        print("=" * 80)

        output_dir = os.path.join('../output', 'EDA')
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Đảm bảo thư mục lưu trữ tồn tại: {output_dir}")

        # 1. Price trends over time
        plt.figure(figsize=(14, 6))
        for ticker in self.df_prices['ticker'].unique():
            ticker_data = self.df_prices[self.df_prices['ticker'] == ticker]
            plt.plot(ticker_data['date'], ticker_data['close'], label=ticker, linewidth=2)
        plt.title('Stock Price Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'price_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

        # 2. Volume analysis
        plt.figure(figsize=(14, 6))
        for ticker in self.df_prices['ticker'].unique():
            ticker_data = self.df_prices[self.df_prices['ticker'] == ticker]
            plt.plot(ticker_data['date'], ticker_data['volume'], label=ticker, alpha=0.7)
        plt.title('Trading Volume Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volume', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'volume_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

        # 3. Distribution of returns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for idx, ticker in enumerate(self.df_prices['ticker'].unique()):
            ticker_data = self.df_prices[self.df_prices['ticker'] == ticker].copy()
            ticker_data['returns'] = ticker_data['close'].pct_change()
            axes[idx].hist(ticker_data['returns'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{ticker} Daily Returns Distribution', fontweight='bold')
            axes[idx].set_xlabel('Returns')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'returns_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

        # 4. Correlation heatmap
        df_pivot = self.df_prices.pivot(index='date', columns='ticker', values='close')
        correlation = df_pivot.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Stock Price Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def preprocessing(self):
        """Clean and preprocess the data"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)

        self.df_clean = self.df_prices.copy()

        # 1. Remove duplicates
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates(subset=['ticker', 'date'])
        removed_dups = initial_rows - len(self.df_clean)
        print(f"\n✓ Removed {removed_dups} duplicate records")

        # 2. Convert date to datetime
        self.df_clean['date'] = pd.to_datetime(self.df_clean['date'])
        print("✓ Converted date to datetime format")

        # 3. Sort by ticker and date
        self.df_clean = self.df_clean.sort_values(['ticker', 'date']).reset_index(drop=True)
        print("✓ Sorted data by ticker and date")

        # 4. Handle missing values (forward fill then backward fill)
        missing_before = self.df_clean.isnull().sum().sum()
        self.df_clean = self.df_clean.fillna(method='ffill').fillna(method='bfill')
        missing_after = self.df_clean.isnull().sum().sum()
        print(f"✓ Handled missing values: {missing_before} → {missing_after}")

        # 5. Feature Engineering
        print("\n📈 FEATURE ENGINEERING")
        print("-" * 80)

        # Daily returns
        self.df_clean['daily_return'] = self.df_clean.groupby('ticker')['close'].pct_change()

        # Moving averages
        self.df_clean['MA_7'] = self.df_clean.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        self.df_clean['MA_30'] = self.df_clean.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean())

        # Volatility (30-day rolling std of returns)
        self.df_clean['volatility'] = self.df_clean.groupby('ticker')['daily_return'].transform(
            lambda x: x.rolling(window=30, min_periods=1).std())

        # Price momentum
        self.df_clean['momentum'] = self.df_clean.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(periods=5))

        # High-Low range
        self.df_clean['price_range'] = self.df_clean['high'] - self.df_clean['low']
        self.df_clean['price_range_pct'] = self.df_clean['price_range'] / self.df_clean['close']

        print("✓ Created daily_return feature")
        print("✓ Created MA_7 (7-day moving average)")
        print("✓ Created MA_30 (30-day moving average)")
        print("✓ Created volatility (30-day rolling std)")
        print("✓ Created momentum (5-day price change)")
        print("✓ Created price_range and price_range_pct")

        # 6. Remove remaining NaN from feature engineering
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna()
        removed_rows = initial_rows - len(self.df_clean)
        print(f"\n✓ Removed {removed_rows} rows with NaN values from feature engineering")

        print(f"\n✅ Final clean dataset shape: {self.df_clean.shape}")
        print(f"\nCleaned data sample:\n{self.df_clean.head(10)}")

        return self.df_clean

    def save_cleaned_data(self):
        """Save cleaned data to CSV and MySQL"""
        # Save to CSV
        self.df_clean.to_csv('stock_data_cleaned.csv', index=False)
        print("\n✓ Saved cleaned data to: stock_data_cleaned.csv")

        # Save to MySQL
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()

            # Create cleaned table
            create_table = """
                           CREATE TABLE IF NOT EXISTS stock_prices_cleaned \
                           ( \
                               id \
                               INT \
                               AUTO_INCREMENT \
                               PRIMARY \
                               KEY, \
                               ticker \
                               VARCHAR \
                           ( \
                               10 \
                           ),
                               date DATE,
                               open DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               high DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               low DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               close DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               volume BIGINT,
                               adj_close DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               daily_return DECIMAL \
                           ( \
                               10, \
                               6 \
                           ),
                               MA_7 DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               MA_30 DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               volatility DECIMAL \
                           ( \
                               10, \
                               6 \
                           ),
                               momentum DECIMAL \
                           ( \
                               10, \
                               6 \
                           ),
                               price_range DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               price_range_pct DECIMAL \
                           ( \
                               10, \
                               6 \
                           ),
                               UNIQUE KEY unique_ticker_date \
                           ( \
                               ticker, \
                               date \
                           )
                               ) \
                           """
            cursor.execute(create_table)

            # Insert data
            for _, row in self.df_clean.iterrows():
                insert_query = """
                               INSERT INTO stock_prices_cleaned
                               (ticker, date, open, high, low, close, volume, adj_close,
                                daily_return, MA_7, MA_30, volatility, momentum, price_range, price_range_pct)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                               UPDATE \
                                   open= \
                               VALUES (open), high= \
                               VALUES (high), low= \
                               VALUES (low), close = \
                               VALUES (close), volume= \
                               VALUES (volume), adj_close= \
                               VALUES (adj_close), daily_return= \
                               VALUES (daily_return), MA_7= \
                               VALUES (MA_7), MA_30= \
                               VALUES (MA_30), volatility= \
                               VALUES (volatility), momentum= \
                               VALUES (momentum), price_range= \
                               VALUES (price_range), price_range_pct= \
                               VALUES (price_range_pct) \
                               """
                values = tuple(row)
                cursor.execute(insert_query, values)

            connection.commit()
            connection.close()
            print("✓ Saved cleaned data to MySQL: stock_prices_cleaned")

        except Exception as e:
            print(f"Error saving to MySQL: {e}")

    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "=" * 80)
        print("🚀 STARTING EDA & PREPROCESSING PIPELINE")
        print("=" * 80)

        self.load_data_from_mysql()
        self.initial_exploration()
        self.data_quality_check()
        self.visualize_data()
        df_clean = self.preprocessing()
        self.save_cleaned_data()

        print("\n" + "=" * 80)
        print("✅ EDA & PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return df_clean

if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'user_name',
        'password': 'your_password',
        'database': 'stock_analysis'
    }

    eda = StockEDA(DB_CONFIG)
    df_cleaned = eda.run_complete_eda()
