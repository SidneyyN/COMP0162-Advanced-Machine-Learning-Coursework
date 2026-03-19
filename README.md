# COMP0162-Advanced-Machine-Learning-Coursework
In this coursework, we are working on a volatility forecasting model using LSTM, and we are comparing it against traditional methods like GARCH. We want to see if an LSTM is able to outperform short-term equity market volatilty forecasting out-of-sample. Our target variable would be the future 5-day realised volatility. 

# Models and Benchmarks
1. Historical Volatility
take the last m returns -> square them -> average them -> square-root -> annualise

We say our volatility of the next m days are going to be his historical volatility.
We are basing it on the past basically.
Baseline model

2. GARCH model
A recursive model
basically it is long-run base level + yesterday shock size + yesterday variance level 
                                        |                       |
                                        |--> big moves tend to be followed by more big moves
                                                                |
                                                                |--> volatility persistence 

At time t, we use GARCH to predict t + 1, t + 2, ..., t + 5 variance, and then we average them and take sqrt 


3. LSTM 

# Plan 

## Phase 1: Data Acquisition & Preprocessing

### Step 1: Data Collection
Download SPY daily OHLCV data from Yahoo Finance (2010-2025)
- Exclude dates with missing data
- Check for splits/dividends adjustments

### Step 2: Data Preprocessing
Calculate daily log returns from adjusted close prices and filter out rows with missing data points.

Create target variable: 5-day realized volatility
- For each date t, calculate volatility of days [t+1, t+5]
- Formula: annualize(sqrt(average(squared log-returns[t+1:t+5])))
- Shift to create proper X/y pairs (no data leakage)

Engineer features for each date t:
- Last 21 log returns (for LSTM input)
- Last 21 squared log returns
- 5-day rolling volatility (using past 5 days)
- 21-day rolling volatility (using past 21 days)
- Exclude any rows where history < 21 days

Split data: Train (2010-2021), Validation (2022-2023), Test (2024-2025)
- Normalize all features using statistics from **training set only**
- Apply same scaler to val/test to prevent data leakage

## Phase 2: Model Implementation

### Step 3: Training the Models

**Historical Volatility baseline**
- Predict: 5-day rolling volatility from training data
- Simple arithmetic average on validation/test sets

**GARCH model**
- Use library (e.g., `statsmodels` GARCH(1,1))
- Fit on training set
- For each val/test date, predict variance for days t+1...t+5
- Average predicted variances and take sqrt to get 5-day vol
- Use MSE as evaluation metric during training

**LSTM model**
- Input: 21-day rolling window (21 × 4 features: returns, squared returns, 5-day vol, 21-day vol)
- Output: single value (predicted 5-day volatility)
- Architecture: flexible (to be determined during development with 1-2 layers as starting point)
- Normalization: apply before feeding to networks
- Train on training set with batch processing
- Use MSE as loss during training; evaluate on validation set
- Early stopping on validation set (monitor MAE or RMSE)

## Phase 3: Hyperparameter Tuning
Determine LSTM hyperparameters through experimentation:
- Layers (1-3), hidden units (32-256), dropout (0-0.5), batch size (32-128)
- Strategy: grid/random search on a subset first
- Evaluate on validation set every epoch

## Phase 4: Evaluation

Evaluate all three models on test set (2024-2025) using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **QLIKE loss**: Specifically measures volatility forecast quality
  - Formula: -1/n * Σ(log(σ²_true) + r²_t/σ²_pred)

Generate comparison:
- Predictions vs actuals for all three models on test set
- Time-series view of forecast errors
- Summary statistics (mean prediction error, bias, etc.)
- Performance comparison table

# Implementation Details

## Data Processing Notes
- Exclude (don't interpolate) missing data dates
- Use 21-day rolling window for LSTM (aligns with 21-day vol feature)
- Normalize at preprocessing stage
- Historical Volatility baseline uses 5-day window

## Key Constraints
- NO DATA LEAKAGE: Rolling window approach, normalize only on training set
- Future volatility uses only past information at each time step
- Train/val/test splits are temporally contiguous (no mixing)

## Files to Create
- `data_collection.py` — download SPY data, handle missing data
- `data_preprocessing.py` — returns, volatility calc, windowing, normalization
- `models.py` — Historical Vol, GARCH, LSTM implementations
- `train.py` — training loop, early stopping, hyperparameter logging
- `evaluate.py` — RMSE, MAE, QLIKE calculations, visualization
- `main.py` or notebook — orchestrates full pipeline