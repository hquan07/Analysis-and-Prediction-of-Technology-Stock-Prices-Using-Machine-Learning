import yfinance as yf
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockETL:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def create_connection(self):
        """Create MySQL database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            if self.connection.is_connected():
                logging.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logging.error(f"Error connecting to MySQL: {e}")
            return False

    def create_tables(self):
        """Create necessary tables in MySQL"""
        try:
            cursor = self.connection.cursor()

            # Create stock_prices table
            create_stock_prices_table = """
                                        CREATE TABLE IF NOT EXISTS stock_prices \
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
                                        ) NOT NULL,
                                            date DATE NOT NULL,
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
                                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                            UNIQUE KEY unique_ticker_date \
                                        ( \
                                            ticker, \
                                            date \
                                        )
                                            ) \
                                        """

            # Create stock_info table
            create_stock_info_table = """
                                      CREATE TABLE IF NOT EXISTS stock_info \
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
                                      ) NOT NULL UNIQUE,
                                          company_name VARCHAR \
                                      ( \
                                          255 \
                                      ),
                                          sector VARCHAR \
                                      ( \
                                          100 \
                                      ),
                                          industry VARCHAR \
                                      ( \
                                          100 \
                                      ),
                                          market_cap BIGINT,
                                          pe_ratio DECIMAL \
                                      ( \
                                          10, \
                                          2 \
                                      ),
                                          dividend_yield DECIMAL \
                                      ( \
                                          5, \
                                          4 \
                                      ),
                                          fifty_two_week_high DECIMAL \
                                      ( \
                                          10, \
                                          2 \
                                      ),
                                          fifty_two_week_low DECIMAL \
                                      ( \
                                          10, \
                                          2 \
                                      ),
                                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                                          ) \
                                      """

            cursor.execute(create_stock_prices_table)
            cursor.execute(create_stock_info_table)
            self.connection.commit()
            logging.info("Tables created successfully")

        except Error as e:
            logging.error(f"Error creating tables: {e}")

    def extract_stock_data(self, tickers, start_date, end_date):
        stock_data = {}

        for ticker in tickers:
            try:
                logging.info(f"Extracting data for {ticker}")
                stock = yf.Ticker(ticker)

                # Get historical data
                df = stock.history(start=start_date, end=end_date)
                df['ticker'] = ticker
                df.reset_index(inplace=True)

                # Get stock info
                info = stock.info

                stock_data[ticker] = {
                    'prices': df,
                    'info': info
                }

                logging.info(f"Successfully extracted {len(df)} records for {ticker}")

            except Exception as e:
                logging.error(f"Error extracting data for {ticker}: {e}")

        return stock_data

    def transform_data(self, stock_data):
        transformed_data = {}

        for ticker, data in stock_data.items():
            try:
                df = data['prices'].copy()

                # Rename columns to match database schema
                df.columns = df.columns.str.lower().str.replace(' ', '_')

                # Convert date to proper format
                df['date'] = pd.to_datetime(df['date']).dt.date

                # Round decimal values
                numeric_cols = ['open', 'high', 'low', 'close', 'adj_close']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].round(2)

                transformed_data[ticker] = {
                    'prices': df,
                    'info': data['info']
                }

                logging.info(f"Transformed data for {ticker}")

            except Exception as e:
                logging.error(f"Error transforming data for {ticker}: {e}")

        return transformed_data

    def load_stock_prices(self, df, ticker):
        """Load stock prices to MySQL"""
        try:
            cursor = self.connection.cursor()

            insert_query = """
                           INSERT INTO stock_prices (ticker, date, open, high, low, close, volume, adj_close)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                           UPDATE \
                               open= \
                           VALUES (open), high= \
                           VALUES (high), low= \
                           VALUES (low), close = \
                           VALUES (close), volume= \
                           VALUES (volume), adj_close= \
                           VALUES (adj_close) \
                           """

            records = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'close']].values.tolist()

            cursor.executemany(insert_query, records)
            self.connection.commit()

            logging.info(f"Loaded {len(records)} price records for {ticker}")

        except Error as e:
            logging.error(f"Error loading stock prices for {ticker}: {e}")

    def load_stock_info(self, info, ticker):
        """Load stock information to MySQL"""
        try:
            cursor = self.connection.cursor()

            insert_query = """
                           INSERT INTO stock_info (ticker, company_name, sector, industry, market_cap,
                                                   pe_ratio, dividend_yield, fifty_two_week_high, fifty_two_week_low)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                           UPDATE \
                               company_name= \
                           VALUES (company_name), sector= \
                           VALUES (sector), industry= \
                           VALUES (industry), market_cap= \
                           VALUES (market_cap), pe_ratio= \
                           VALUES (pe_ratio), dividend_yield= \
                           VALUES (dividend_yield), fifty_two_week_high= \
                           VALUES (fifty_two_week_high), fifty_two_week_low= \
                           VALUES (fifty_two_week_low) \
                           """

            record = (
                ticker,
                info.get('longName', None),
                info.get('sector', None),
                info.get('industry', None),
                info.get('marketCap', None),
                info.get('trailingPE', None),
                info.get('dividendYield', None),
                info.get('fiftyTwoWeekHigh', None),
                info.get('fiftyTwoWeekLow', None)
            )

            cursor.execute(insert_query, record)
            self.connection.commit()

            logging.info(f"Loaded info for {ticker}")

        except Error as e:
            logging.error(f"Error loading stock info for {ticker}: {e}")

    def run_etl(self, tickers, start_date, end_date):
        logging.info("Starting ETL Pipeline")

        # Connect to database
        if not self.create_connection():
            logging.error("Failed to connect to database. Exiting.")
            return

        # Create tables
        self.create_tables()

        # Extract
        logging.info("EXTRACT phase started")
        stock_data = self.extract_stock_data(tickers, start_date, end_date)

        # Transform
        logging.info("TRANSFORM phase started")
        transformed_data = self.transform_data(stock_data)

        # Load
        logging.info("LOAD phase started")
        for ticker, data in transformed_data.items():
            self.load_stock_prices(data['prices'], ticker)
            self.load_stock_info(data['info'], ticker)

        logging.info("ETL Pipeline completed successfully")

        # Close connection
        if self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed")

if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'user_name',
        'password': 'your_password',
        'database': 'stock_analysis'
    }

    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    etl = StockETL(DB_CONFIG)
    etl.run_etl(TICKERS, START_DATE, END_DATE)
