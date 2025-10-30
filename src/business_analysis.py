import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from datetime import datetime, timedelta
import os

class BusinessAnalysis:
    def __init__(self, db_config):
        """Initialize Business Analysis"""
        self.db_config = db_config
        self.df = None
        self.df_info = None
        script_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(script_path)
        self.project_root = os.path.dirname(src_dir)
        self.output_dir = os.path.join(self.project_root, 'output')
        self.ba_output_dir = os.path.join(self.output_dir, 'BA')

    def load_cleaned_data(self):
        """Load cleaned data from MySQL"""
        try:
            connection = mysql.connector.connect(**self.db_config)

            # Load cleaned prices
            query = """
                    SELECT * \
                    FROM stock_prices_cleaned
                    ORDER BY ticker, date \
                    """
            self.df = pd.read_sql(query, connection)
            self.df['date'] = pd.to_datetime(self.df['date'])

            # Load stock info
            query_info = """
                         SELECT * \
                         FROM stock_info \
                         """
            self.df_info = pd.read_sql(query_info, connection)

            connection.close()

            print(f"✓ Loaded {len(self.df)} records for analysis")

        except Exception as e:
            print(f"Error loading data: {e}")

    def performance_analysis(self):
        """Analyze stock performance metrics"""
        print("\n" + "=" * 80)
        print("📊 STOCK PERFORMANCE ANALYSIS")
        print("=" * 80)

        results = []

        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker].copy()

            # Calculate metrics
            start_price = ticker_data.iloc[0]['close']
            end_price = ticker_data.iloc[-1]['close']
            total_return = ((end_price - start_price) / start_price) * 100

            avg_daily_return = ticker_data['daily_return'].mean() * 100
            volatility = ticker_data['daily_return'].std() * 100

            # Sharpe Ratio (assuming risk-free rate = 2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            sharpe_ratio = (ticker_data['daily_return'].mean() - risk_free_rate) / ticker_data['daily_return'].std()
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(252)

            # Max drawdown
            cumulative = (1 + ticker_data['daily_return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            # Trading volume metrics
            avg_volume = ticker_data['volume'].mean()
            total_volume = ticker_data['volume'].sum()

            results.append({
                'Ticker': ticker,
                'Start Price': f"${start_price:.2f}",
                'End Price': f"${end_price:.2f}",
                'Total Return (%)': f"{total_return:.2f}%",
                'Avg Daily Return (%)': f"{avg_daily_return:.4f}%",
                'Volatility (%)': f"{volatility:.2f}%",
                'Sharpe Ratio': f"{sharpe_ratio_annual:.2f}",
                'Max Drawdown (%)': f"{max_drawdown:.2f}%",
                'Avg Volume': f"{avg_volume:,.0f}",
                'Risk/Return': volatility / abs(avg_daily_return) if avg_daily_return != 0 else np.inf
            })

        df_performance = pd.DataFrame(results)
        print("\n", df_performance.to_string(index=False))

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Total Returns comparison
        tickers = df_performance['Ticker'].values
        returns = [float(r.strip('%')) for r in df_performance['Total Return (%)']]
        colors = ['green' if r > 0 else 'red' for r in returns]

        axes[0, 0].bar(tickers, returns, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)

        # 2. Risk-Return scatter
        volatilities = [float(v.strip('%')) for v in df_performance['Volatility (%)']]
        avg_returns = [float(r.strip('%')) for r in df_performance['Avg Daily Return (%)']]

        axes[0, 1].scatter(volatilities, avg_returns, s=200, alpha=0.6, c=colors, edgecolors='black')
        for i, ticker in enumerate(tickers):
            axes[0, 1].annotate(ticker, (volatilities[i], avg_returns[i]),
                                fontsize=10, fontweight='bold')
        axes[0, 1].set_xlabel('Volatility (%)')
        axes[0, 1].set_ylabel('Avg Daily Return (%)')
        axes[0, 1].set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sharpe Ratio comparison
        sharpe_ratios = [float(sr) for sr in df_performance['Sharpe Ratio']]
        axes[1, 0].barh(tickers, sharpe_ratios, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)

        # 4. Trading Volume
        avg_volumes = [float(v.replace(',', '')) for v in df_performance['Avg Volume']]
        axes[1, 1].bar(tickers, avg_volumes, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Average Trading Volume', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='plain', axis='y')

        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: performance_analysis.png")
        os.makedirs(self.ba_output_dir, exist_ok=True)
        save_path = os.path.join(self.ba_output_dir, 'performance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
        plt.close()

        return df_performance

    def trend_analysis(self):
        """Analyze price trends and patterns"""
        print("\n" + "=" * 80)
        print("📈 TREND ANALYSIS")
        print("=" * 80)

        fig, axes = plt.subplots(len(self.df['ticker'].unique()), 1,
                                 figsize=(14, 4 * len(self.df['ticker'].unique())))

        if len(self.df['ticker'].unique()) == 1:
            axes = [axes]

        for idx, ticker in enumerate(self.df['ticker'].unique()):
            ticker_data = self.df[self.df['ticker'] == ticker].copy()

            # Plot price and moving averages
            axes[idx].plot(ticker_data['date'], ticker_data['close'],
                           label='Close Price', linewidth=2, color='blue')
            axes[idx].plot(ticker_data['date'], ticker_data['MA_7'],
                           label='7-day MA', linewidth=1.5, color='orange', linestyle='--')
            axes[idx].plot(ticker_data['date'], ticker_data['MA_30'],
                           label='30-day MA', linewidth=1.5, color='red', linestyle='--')

            # Identify trend
            recent_data = ticker_data.tail(30)
            trend = "UPTREND" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "DOWNTREND"
            trend_color = 'green' if trend == "UPTREND" else 'red'

            axes[idx].set_title(f'{ticker} - Price Trend Analysis (Current: {trend})',
                                fontsize=12, fontweight='bold', color=trend_color)
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Price ($)')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)

            # Print trend analysis
            print(f"\n{ticker}:")
            print(f"  Current Trend: {trend}")
            print(f"  Price above 7-day MA: {ticker_data['close'].iloc[-1] > ticker_data['MA_7'].iloc[-1]}")
            print(f"  Price above 30-day MA: {ticker_data['close'].iloc[-1] > ticker_data['MA_30'].iloc[-1]}")
            print(f"  Current Volatility: {ticker_data['volatility'].iloc[-1]:.4f}")

        plt.tight_layout()
        plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: trend_analysis.png")
        os.makedirs(self.ba_output_dir, exist_ok=True)
        save_path = os.path.join(self.ba_output_dir, 'trend_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
        plt.close()

    def volatility_analysis(self):
        """Analyze volatility patterns"""
        print("\n" + "=" * 80)
        print("📊 VOLATILITY ANALYSIS")
        print("=" * 80)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Volatility over time
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker]
            axes[0].plot(ticker_data['date'], ticker_data['volatility'],
                         label=ticker, linewidth=1.5, alpha=0.8)

        axes[0].set_title('Volatility Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Volatility')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Average volatility comparison
        avg_volatility = []
        tickers = []
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker]
            avg_vol = ticker_data['volatility'].mean()
            avg_volatility.append(avg_vol)
            tickers.append(ticker)
            print(f"{ticker} Average Volatility: {avg_vol:.6f}")

        axes[1].bar(tickers, avg_volatility, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_title('Average Volatility Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Volatility')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: volatility_analysis.png")
        os.makedirs(self.ba_output_dir, exist_ok=True)
        save_path = os.path.join(self.ba_output_dir, 'volatility_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
        plt.close()

    def sector_analysis(self):
        """Analyze by sector and industry"""
        print("\n" + "=" * 80)
        print("🏢 SECTOR & INDUSTRY ANALYSIS")
        print("=" * 80)

        if self.df_info is not None:
            print("\n", self.df_info[['ticker', 'company_name', 'sector', 'industry']].to_string(index=False))

            # Market cap analysis
            if 'market_cap' in self.df_info.columns:
                print("\n💰 MARKET CAPITALIZATION")
                print("-" * 80)
                for _, row in self.df_info.iterrows():
                    if pd.notna(row['market_cap']):
                        market_cap_b = row['market_cap'] / 1e9
                        print(f"{row['ticker']}: ${market_cap_b:.2f}B")

    def correlation_analysis(self):
        """Analyze correlations between stocks"""
        print("\n" + "=" * 80)
        print("🔗 CORRELATION ANALYSIS")
        print("=" * 80)

        # Create pivot table for returns
        df_pivot = self.df.pivot(index='date', columns='ticker', values='daily_return')
        correlation = df_pivot.corr()

        print("\nReturn Correlation Matrix:")
        print(correlation.to_string())

        # Find highly correlated pairs
        print("\n🔍 Highly Correlated Stock Pairs (> 0.8):")
        for i in range(len(correlation.columns)):
            for j in range(i + 1, len(correlation.columns)):
                if abs(correlation.iloc[i, j]) > 0.8:
                    print(f"{correlation.columns[i]} - {correlation.columns[j]}: {correlation.iloc[i, j]:.3f}")

    def generate_insights(self):
        """Generate business insights"""
        print("\n" + "=" * 80)
        print("💡 KEY BUSINESS INSIGHTS")
        print("=" * 80)

        insights = []

        # Best performing stock
        best_returns = {}
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker]
            total_return = ((ticker_data.iloc[-1]['close'] - ticker_data.iloc[0]['close']) /
                            ticker_data.iloc[0]['close']) * 100
            best_returns[ticker] = total_return

        best_stock = max(best_returns, key=best_returns.get)
        worst_stock = min(best_returns, key=best_returns.get)

        insights.append(f"1. BEST PERFORMER: {best_stock} with {best_returns[best_stock]:.2f}% return")
        insights.append(f"2. WORST PERFORMER: {worst_stock} with {best_returns[worst_stock]:.2f}% return")

        # Most volatile
        volatilities = {}
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker]
            volatilities[ticker] = ticker_data['volatility'].mean()

        most_volatile = max(volatilities, key=volatilities.get)
        least_volatile = min(volatilities, key=volatilities.get)

        insights.append(f"3. HIGHEST VOLATILITY: {most_volatile}")
        insights.append(f"4. LOWEST VOLATILITY: {least_volatile}")

        # Trading activity
        volumes = {}
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker]
            volumes[ticker] = ticker_data['volume'].mean()

        most_traded = max(volumes, key=volumes.get)
        insights.append(f"5. MOST TRADED: {most_traded} with avg volume {volumes[most_traded]:,.0f}")

        for insight in insights:
            print(f"\n{insight}")

        # Save insights to file
        with open('../output/BA/business_insights.txt', 'w') as f:
            f.write("BUSINESS INSIGHTS - STOCK ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            for insight in insights:
                f.write(insight + "\n")

        for insight in insights:
            print(f"\n{insight}")

            # Save insights to file
        os.makedirs(self.ba_output_dir, exist_ok=True)
        save_path = os.path.join(self.ba_output_dir, 'business_insights.txt')
        with open(save_path, 'w') as f:
            f.write("BUSINESS INSIGHTS - STOCK ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            for insight in insights:
                f.write(insight + "\n")
        print(f"\n✓ Saved: {save_path}")

        print("\n✓ Saved: business_insights.txt")


    def run_analysis(self):
        """Run complete business analysis"""
        print("\n" + "=" * 80)
        print("🚀 STARTING BUSINESS ANALYSIS")
        print("=" * 80)

        self.load_cleaned_data()
        self.performance_analysis()
        self.trend_analysis()
        self.volatility_analysis()
        self.sector_analysis()
        self.correlation_analysis()
        self.generate_insights()

        print("\n" + "=" * 80)
        print("✅ BUSINESS ANALYSIS COMPLETED")
        print("=" * 80)


# Main execution
if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Huyquan160720040@',
        'database': 'stock_analysis'
    }

    analysis = BusinessAnalysis(DB_CONFIG)
    analysis.run_analysis()