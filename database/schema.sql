-- Create database
CREATE DATABASE IF NOT EXISTS stock_analysis;
USE stock_analysis;

-- 1. Stock Prices Table (Raw data from yfinance)
CREATE TABLE IF NOT EXISTS stock_prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    adj_close DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_ticker_date (ticker, date),
    INDEX idx_ticker (ticker),
    INDEX idx_date (date),
    INDEX idx_ticker_date (ticker, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Stock Information Table
CREATE TABLE IF NOT EXISTS stock_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    pe_ratio DECIMAL(10, 2),
    dividend_yield DECIMAL(5, 4),
    fifty_two_week_high DECIMAL(10, 2),
    fifty_two_week_low DECIMAL(10, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_ticker (ticker),
    INDEX idx_sector (sector),
    INDEX idx_industry (industry)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Stock Prices Cleaned Table (After preprocessing)
CREATE TABLE IF NOT EXISTS stock_prices_cleaned (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    adj_close DECIMAL(10, 2),

    -- Engineered features
    daily_return DECIMAL(10, 6),
    MA_7 DECIMAL(10, 2),
    MA_30 DECIMAL(10, 2),
    volatility DECIMAL(10, 6),
    momentum DECIMAL(10, 6),
    price_range DECIMAL(10, 2),
    price_range_pct DECIMAL(10, 6),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_ticker_date (ticker, date),
    INDEX idx_ticker (ticker),
    INDEX idx_date (date),
    INDEX idx_ticker_date (ticker, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Model Performance Table (ML results)
CREATE TABLE IF NOT EXISTS model_comparison (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,

    -- Dùng DOUBLE cho tất cả các cột metric --
    train_rmse DOUBLE,
    test_rmse DOUBLE,
    train_mae DOUBLE,
    test_mae DOUBLE,
    train_r2 DOUBLE,
    test_r2 DOUBLE,
    train_mape DOUBLE,
    test_mape DOUBLE,
    overfitting_score DOUBLE,

    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_ticker_model (ticker, model_name, training_date),
    INDEX idx_ticker (ticker),
    INDEX idx_model (model_name),
    INDEX idx_test_r2 (test_r2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Useful Queries

-- Get latest prices for all stocks
-- SELECT ticker, date, close, volume
-- FROM stock_prices
-- WHERE date = (SELECT MAX(date) FROM stock_prices)
-- ORDER BY ticker;

-- Get stock performance summary
-- SELECT
--     ticker,
--     MIN(close) as min_price,
--     MAX(close) as max_price,
--     AVG(close) as avg_price,
--     AVG(volume) as avg_volume
-- FROM stock_prices
-- GROUP BY ticker;

-- Get best performing stocks by return
-- SELECT ticker, total_return, sharpe_ratio
-- FROM business_insights
-- WHERE analysis_date = (SELECT MAX(analysis_date) FROM business_insights)
-- ORDER BY total_return DESC;

-- Get best ML models by R² score
-- SELECT ticker, model_name, test_r2, test_rmse, test_mape
-- FROM model_performance
-- WHERE training_date = (SELECT MAX(training_date) FROM model_performance)
-- ORDER BY test_r2 DESC;

-- ============================================================================
-- Data Maintenance
-- ============================================================================

-- Clean old data (older than 5 years)
-- DELETE FROM stock_prices WHERE date < DATE_SUB(CURDATE(), INTERVAL 5 YEAR);

-- Vacuum tables to reclaim space
-- OPTIMIZE TABLE stock_prices;
-- OPTIMIZE TABLE stock_prices_cleaned;
-- OPTIMIZE TABLE model_performance;