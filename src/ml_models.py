import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

class StockMLModels:
    def __init__(self, db_config):
        """Initialize ML Models"""
        self.db_config = db_config
        self.df = None
        self.models = {}
        self.results = []
        self.scaler = StandardScaler()

    def load_data(self):
        """Load cleaned data from MySQL"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            query = "SELECT * FROM stock_prices_cleaned ORDER BY ticker, date"
            self.df = pd.read_sql(query, connection)
            self.df['date'] = pd.to_datetime(self.df['date'])
            connection.close()
            print(f"✓ Loaded {len(self.df)} records")
        except Exception as e:
            print(f"Error loading data: {e}")

    def prepare_features(self, ticker):
        """Prepare features for ML models"""
        print(f"\n📊 Preparing features for {ticker}...")

        ticker_data = self.df[self.df['ticker'] == ticker].copy().reset_index(drop=True)

        # Additional feature engineering
        ticker_data['day_of_week'] = ticker_data['date'].dt.dayofweek
        ticker_data['month'] = ticker_data['date'].dt.month
        ticker_data['quarter'] = ticker_data['date'].dt.quarter

        # Lag features
        for lag in [1, 2, 3, 5, 7]:
            ticker_data[f'close_lag_{lag}'] = ticker_data['close'].shift(lag)
            ticker_data[f'volume_lag_{lag}'] = ticker_data['volume'].shift(lag)

        # Rolling statistics
        ticker_data['close_rolling_mean_7'] = ticker_data['close'].rolling(window=7).mean()
        ticker_data['close_rolling_std_7'] = ticker_data['close'].rolling(window=7).std()
        ticker_data['volume_rolling_mean_7'] = ticker_data['volume'].rolling(window=7).mean()

        # Drop rows with NaN
        ticker_data = ticker_data.dropna()

        # Select features
        feature_columns = [
            'open', 'high', 'low', 'volume',
            'MA_7', 'MA_30', 'volatility', 'momentum',
            'price_range', 'price_range_pct',
            'day_of_week', 'month', 'quarter',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_7',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_7',
            'close_rolling_mean_7', 'close_rolling_std_7', 'volume_rolling_mean_7'
        ]

        X = ticker_data[feature_columns]
        y = ticker_data['close']  # Target: predict closing price

        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")

        return X, y, ticker_data

    def split_and_scale_data(self, X, y):
        """Split data and scale features"""
        # Time series split (keep temporal order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"✓ Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, X_test, y_train, y_test, ticker):
        """Train multiple ML models"""
        print(f"\n🤖 Training models for {ticker}...")

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }

        ticker_results = []

        for name, model in models.items():
            print(f"\n  Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Calculate MAPE
            train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
            test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

            result = {
                'Ticker': ticker,
                'Model': name,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train MAPE (%)': train_mape,
                'Test MAPE (%)': test_mape,
                'Overfitting': abs(train_r2 - test_r2)
            }

            ticker_results.append(result)

            # Store model and predictions
            self.models[f"{ticker}_{name}"] = {
                'model': model,
                'predictions': y_test_pred,
                'actual': y_test
            }

            print(f"    Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}, Test MAPE: {test_mape:.2f}%")

        return ticker_results

    def compare_models(self):
        """Create comprehensive model comparison"""
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON RESULTS")
        print("=" * 80)

        df_results = pd.DataFrame(self.results)

        # Display full results
        print("\n" + "=" * 120)
        print("DETAILED MODEL PERFORMANCE")
        print("=" * 120)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df_results.to_string(index=False))

        # Find best model for each ticker
        print("\n" + "=" * 80)
        print("🏆 BEST MODELS BY TICKER")
        print("=" * 80)

        for ticker in df_results['Ticker'].unique():
            ticker_results = df_results[df_results['Ticker'] == ticker]
            best_model = ticker_results.loc[ticker_results['Test R²'].idxmax()]
            print(f"\n{ticker}:")
            print(f"  Best Model: {best_model['Model']}")
            print(f"  Test R²: {best_model['Test R²']:.4f}")
            print(f"  Test RMSE: {best_model['Test RMSE']:.4f}")
            print(f"  Test MAPE: {best_model['Test MAPE (%)']:.2f}%")

        # Overall best model
        best_overall = df_results.loc[df_results['Test R²'].idxmax()]
        print("\n" + "=" * 80)
        print("🥇 OVERALL BEST MODEL")
        print("=" * 80)
        print(f"Ticker: {best_overall['Ticker']}")
        print(f"Model: {best_overall['Model']}")
        print(f"Test R²: {best_overall['Test R²']:.4f}")
        print(f"Test RMSE: {best_overall['Test RMSE']:.4f}")
        print(f"Test MAPE: {best_overall['Test MAPE (%)']:.2f}%")

        # Save results to CSV
        df_results.to_csv('model_comparison.csv', index=False)
        print("\n✓ Saved: model_comparison.csv")

        return df_results

    def visualize_results(self, df_results):
        """Create visualizations for model comparison"""
        print("\n📊 Creating visualizations...")

        # 1. Test R² Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R² Score comparison
        pivot_r2 = df_results.pivot(index='Model', columns='Ticker', values='Test R²')
        pivot_r2.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Test R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].legend(title='Ticker', bbox_to_anchor=(1.05, 1))
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.8, color='green', linestyle='--', label='Good threshold', alpha=0.5)

        # RMSE comparison
        pivot_rmse = df_results.pivot(index='Model', columns='Ticker', values='Test RMSE')
        pivot_rmse.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].legend(title='Ticker', bbox_to_anchor=(1.05, 1))
        axes[0, 1].grid(True, alpha=0.3)

        # MAPE comparison
        pivot_mape = df_results.pivot(index='Model', columns='Ticker', values='Test MAPE (%)')
        pivot_mape.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Test MAPE Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].legend(title='Ticker', bbox_to_anchor=(1.05, 1))
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting analysis
        pivot_overfit = df_results.pivot(index='Model', columns='Ticker', values='Overfitting')
        pivot_overfit.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title('Overfitting Analysis (|Train R² - Test R²|)',
                             fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Overfitting Score')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].legend(title='Ticker', bbox_to_anchor=(1.05, 1))
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.1, color='red', linestyle='--',
                           label='High overfitting threshold', alpha=0.5)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison.png")
        plt.close()

        # 2. Prediction vs Actual plots
        for ticker in df_results['Ticker'].unique():
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.ravel()

            ticker_models = df_results[df_results['Ticker'] == ticker]

            for idx, (_, row) in enumerate(ticker_models.iterrows()):
                model_key = f"{ticker}_{row['Model']}"
                if model_key in self.models:
                    pred = self.models[model_key]['predictions']
                    actual = self.models[model_key]['actual']

                    axes[idx].scatter(actual, pred, alpha=0.5, s=10)
                    axes[idx].plot([actual.min(), actual.max()],
                                   [actual.min(), actual.max()],
                                   'r--', lw=2)
                    axes[idx].set_title(f"{row['Model']}\nR²={row['Test R²']:.3f}",
                                        fontsize=10, fontweight='bold')
                    axes[idx].set_xlabel('Actual Price')
                    axes[idx].set_ylabel('Predicted Price')
                    axes[idx].grid(True, alpha=0.3)

            plt.suptitle(f'{ticker} - Predictions vs Actual Prices',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'predictions_{ticker}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: predictions_{ticker}.png")
            plt.close()

    def generate_insights(self, df_results):
        """Generate ML insights"""
        print("\n" + "=" * 80)
        print("💡 MACHINE LEARNING INSIGHTS")
        print("=" * 80)

        insights = []

        # Best performing model overall
        best_model = df_results.loc[df_results['Test R²'].idxmax()]
        insights.append(f"1. BEST OVERALL MODEL: {best_model['Model']} "
                        f"({best_model['Ticker']}) with R²={best_model['Test R²']:.4f}")

        # Most consistent model across tickers
        model_avg_r2 = df_results.groupby('Model')['Test R²'].mean().sort_values(ascending=False)
        insights.append(f"2. MOST CONSISTENT MODEL: {model_avg_r2.index[0]} "
                        f"with avg R²={model_avg_r2.values[0]:.4f}")

        # Least overfitting model
        least_overfit = df_results.loc[df_results['Overfitting'].idxmin()]
        insights.append(f"3. LEAST OVERFITTING: {least_overfit['Model']} "
                        f"({least_overfit['Ticker']}) with {least_overfit['Overfitting']:.4f}")

        # Best accuracy (lowest MAPE)
        best_accuracy = df_results.loc[df_results['Test MAPE (%)'].idxmin()]
        insights.append(f"4. HIGHEST ACCURACY: {best_accuracy['Model']} "
                        f"({best_accuracy['Ticker']}) with MAPE={best_accuracy['Test MAPE (%)']:.2f}%")

        # Model recommendations
        insights.append("\n5. RECOMMENDATIONS:")
        insights.append("   - For production: Choose models with R² > 0.85 and low overfitting")
        insights.append("   - For stability: Prefer ensemble methods (Random Forest, XGBoost)")
        insights.append("   - For interpretability: Consider Linear/Ridge/Lasso if performance is acceptable")

        for insight in insights:
            print(insight)

        # Save insights
        with open('../output/ml_models/ml_insights.txt', 'w') as f:
            f.write("MACHINE LEARNING INSIGHTS\n")
            f.write("=" * 80 + "\n\n")
            for insight in insights:
                f.write(insight + "\n")

        print("\n✓ Saved: ml_insights.txt")

    def save_results_to_mysql(self, df_results):
        print("\n💾 Save result moedl comaprison to database MySQL...")
        df_to_save = df_results.rename(columns={
            'Ticker': 'ticker',
            'Model': 'model_name',
            'Train RMSE': 'train_rmse',
            'Test RMSE': 'test_rmse',
            'Train MAE': 'train_mae',
            'Test MAE': 'test_mae',
            'Train R²': 'train_r2',
            'Test R²': 'test_r2',
            'Train MAPE (%)': 'train_mape',
            'Test MAPE (%)': 'test_mape',
            'Overfitting': 'overfitting_score'
        })

        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()

            insert_query = """
                           INSERT INTO model_comparison (ticker, model_name, train_rmse, test_rmse, train_mae, test_mae, \
                                                         train_r2, test_r2, train_mape, test_mape, overfitting_score) \
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) \
                           """

            columns_to_insert = [
                'ticker', 'model_name', 'train_rmse', 'test_rmse', 'train_mae', 'test_mae',
                'train_r2', 'test_r2', 'train_mape', 'test_mape', 'overfitting_score'
            ]

            records = [tuple(r) for r in df_to_save[columns_to_insert].to_numpy()]

            cursor.executemany(insert_query, records)
            connection.commit()

            print(f"✓ Save successfully {len(records)} result to table 'model_comparison'.")

        except Exception as e:
            print(f"Error saved to MySQL: {e}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

    def run_complete_ml_pipeline(self, tickers=None):
        """Run complete ML pipeline"""
        print("\n" + "=" * 80)
        print("🚀 STARTING MACHINE LEARNING PIPELINE")
        print("=" * 80)

        self.load_data()

        if tickers is None:
            tickers = self.df['ticker'].unique()

        # Train models for each ticker
        for ticker in tickers:
            print(f"\n{'=' * 80}")
            print(f"Processing {ticker}")
            print('=' * 80)

            X, y, ticker_data = self.prepare_features(ticker)
            X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
            ticker_results = self.train_models(X_train, X_test, y_train, y_test, ticker)
            self.results.extend(ticker_results)

        # Compare and visualize
        df_results = self.compare_models()
        self.visualize_results(df_results)
        self.generate_insights(df_results)

        print("\n" + "=" * 80)
        print("✅ MACHINE LEARNING PIPELINE COMPLETED")
        print("=" * 80)

        self.save_results_to_mysql(df_results)
        print("\n" + "=" * 80)
        print("✅ MACHINE LEARNING PIPELINE COMPLETED")

        return df_results

if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'user_name',
        'password': 'your_password',
        'database': 'stock_analysis'
    }

    ml = StockMLModels(DB_CONFIG)
    results = ml.run_complete_ml_pipeline()
