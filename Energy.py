# Installing Prophet (if needed)
!pip install prophet --quiet

# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import statsmodels.api as sm
from prophet import Prophet
import xgboost as xgb
import holidays
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error
from prophet.plot import plot_plotly
warnings.filterwarnings("ignore")

df = pd.read_csv('/kaggle/input/smart-meter-electricity-consumption-dataset/smart_meter_data.csv')
print(df.columns)

'''
Output:

Index(['Timestamp', 'Electricity_Consumed', 'Temperature', 'Humidity',
       'Wind_Speed', 'Avg_Past_Consumption', 'Anomaly_Label'],
      dtype='object')
'''

# Converting 'Timestamp' column to datetime type
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Setting 'Timestamp' as the dataframe index
df = df.set_index('Timestamp')
# Sorting index 
df = df.sort_index()
print(df.index)
print(df.head())

# Renaming columns for clarity
df.rename(columns={'Electricity_Consumed': 'consumption',
                   'Temperature': 'temperature',
                   'Humidity': 'humidity',
                   'Wind_Speed': 'wind_speed',
                   'Avg_Past_Consumption': 'avg_past_consumption',
                   'Anomaly_Label': 'anomaly_label'}, inplace=True)

# Resampling numeric columns with mean + interpolate
df_numeric = df.select_dtypes(include=['number'])
df_num_resampled = df_numeric.resample('H').mean().interpolate()

# Resampling non-numeric columns with forward fill (or appropriate method)
df_non_numeric = df.select_dtypes(exclude=['number'])
df_non_num_resampled = df_non_numeric.resample('H').ffill()

# Combing back
df = pd.concat([df_num_resampled, df_non_num_resampled], axis=1)

print(df.head())
print(df.dtypes)


'''
Output

consumption  temperature  humidity  wind_speed  \
Timestamp                                                             
2024-01-01 00:00:00     0.404871     0.467535  0.423776    0.452085   
2024-01-01 01:00:00     0.555893     0.383755  0.460299    0.523301   
2024-01-01 02:00:00     0.335976     0.490065  0.527394    0.367058   
2024-01-01 03:00:00     0.570554     0.629478  0.512184    0.509074   
2024-01-01 04:00:00     0.381090     0.467629  0.493948    0.357784   

                     avg_past_consumption anomaly_label  
Timestamp                                                
2024-01-01 00:00:00              0.615965        Normal  
2024-01-01 01:00:00              0.685884        Normal  
2024-01-01 02:00:00              0.646294        Normal  
2024-01-01 03:00:00              0.710637        Normal  
2024-01-01 04:00:00              0.666220        Normal  
consumption             float64
temperature             float64
humidity                float64
wind_speed              float64
avg_past_consumption    float64
anomaly_label            object
dtype: object

'''

# Dropping the anomaly_label (object) column entirely as it is not needed further. 
df = df.drop(columns=['anomaly_label'])



## Feature Engineering ##

# Using Canadian holidays
canada_holidays = holidays.CA()
# Note: Holidays can drastically affect electricity demand. As on holidays: Offices are closed, homes are occupied → load profile changes

# Calendar-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_holiday'] = df.index.to_series().apply(lambda x: 1 if x in canada_holidays else 0)

'''
In the above function of df['is_holiday']. 
i.)  .to_series()	---   Converts the datetime index to a pandas Series so that we can apply functions row by row.
ii.) .apply(...)	---   Applies a function to each timestamp in the Series — this function checks if the timestamp is a holiday.
iii.) lambda x: 1 if x in canada_holidays else 0 ---  Short inline function (lambda) that says: “If x (the datetime) is found in the canada_holidays calendar, return 1, else return 0."
iv.)  canada_holidays = holidays.CA() ---  Creates a Python holiday calendar using the holidays library
'''

# Lag features
df['lag_1'] = df['consumption'].shift(1)
df['lag_24'] = df['consumption'].shift(24)

'''
.shift(n) - It moves the data down by n time steps, effectively giving you the value from n steps ago. Allowing the model to "look back" at what happened in the past and learn how the past affects the future.

lag_1: value of consumption 1 hour ago
lag_24: value of consumption 24 hours ago (i.e., same hour yesterday)
'''

# Rolling statistics
df['rolling_mean_24'] = df['consumption'].rolling(window=24).mean()
'''
.rolling(window=24).mean() - It calculates the average of the last 24 values (i.e., last 24 hours). A moving average —  slides over the time series and smooths it
'''

# Dropping any resulting NaN rows due to lag/rolling
df.dropna(inplace=True)


## Forecasting Model 1: ARIMA (AutoRegressive Integrated Moving Average) ##

# Target variable
ts = df['consumption']

# Split point (Using the last 20% of the data as a test set)
split_index = int(len(ts) * 0.8)
train, test = ts[:split_index], ts[split_index:]

# Fit SARIMAX (p,d,q)(P,D,Q,s) - these values can be tuned (Using SARIMAX (ARIMA with exogenous variables and seasonal handling))
model = sm.tsa.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,24))
'''
SARIMAX stands for Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors.

It’s a powerful time series model that can capture:
• Trends (using integration/differencing)
• Autoregressive relationships (dependence on past values)
• Moving average components (dependence on past errors)
• Seasonality (regular repeating patterns, e.g., daily or weekly)
• Exogenous variables (additional external features, if provided)

Parameters:
a.) train - Training data series

b.) order - (1, 1, 1) - The non-seasonal ARIMA part
Breakdown of order=(p, d, q):
• p = 1: Number of autoregressive (AR) terms — model uses one lagged value to predict current value.
• d = 1: Number of differences needed to make the series stationary (removes trend).
• q = 1: Number of moving average (MA) terms — model uses one lagged forecast error.

c.) seasonal_order - (1, 1, 1, 24) - The seasonal ARIMA part
Breakdown of seasonal_order=(P, D, Q, s):
• P = 1: Number of seasonal autoregressive terms — captures repeating patterns at seasonal lag.
• D = 1: Number of seasonal differences — removes seasonal trend.
• Q = 1: Number of seasonal moving average terms — smooths seasonal noise.
• s = 24: The length of the seasonal cycle — here 24 means daily seasonality (24 hours per day).
SARIMAX stands for Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors.

It’s a powerful time series model that can capture:
• Trends (using integration/differencing)
• Autoregressive relationships (dependence on past values)
• Moving average components (dependence on past errors)
• Seasonality (regular repeating patterns, e.g., daily or weekly)
• Exogenous variables (additional external features, if provided)

Parameters:
a.) train - Training data series

b.) order - (1, 1, 1) - The non-seasonal ARIMA part
Breakdown of order=(p, d, q):
• p = 1: Number of autoregressive (AR) terms — model uses one lagged value to predict current value.
• d = 1: Number of differences needed to make the series stationary (removes trend).
• q = 1: Number of moving average (MA) terms — model uses one lagged forecast error.

c.) seasonal_order - (1, 1, 1, 24) - The seasonal ARIMA part
Breakdown of seasonal_order=(P, D, Q, s):
• P = 1: Number of seasonal autoregressive terms — captures repeating patterns at seasonal lag.
• D = 1: Number of seasonal differences — removes seasonal trend.
• Q = 1: Number of seasonal moving average terms — smooths seasonal noise.
• s = 24: The length of the seasonal cycle — here 24 means daily seasonality (24 hours per day).
'''

model_fit = model.fit(disp=False)
'''
.fit() trains the SARIMAX model on your training data, estimating parameters. disp=False turns off detailed output during fitting to keep console clean.
'''

# Forecast for the test period
forecast = model_fit.forecast(steps=len(test))
'''
.forecast() asks the model to predict future values beyond the training data. steps=len(test) tells it how many time points to predict into the future — exactly the length of your test dataset.
'''

# Evaluate
rmse_arima = np.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA RMSE: {rmse_arima:.4f}")

'''
Output: 
ARIMA RMSE: 0.1143
'''

# Plotting visually

plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='gray')
plt.plot(test.index, forecast, label='Forecast (ARIMA)', color='green')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()


## Forecasting Model 2: Prophet ##

# Resetting index to move Timestamp into a column
df_prophet = df[['consumption']].reset_index()

# Renaming columns as Prophet expects
df_prophet = df_prophet.rename(columns={'Timestamp': 'ds', 'consumption': 'y'})

print(df_prophet.head())

# Splitting as before (80/20):
split_index = int(len(df_prophet) * 0.8)
train_prophet = df_prophet.iloc[:split_index]
test_prophet = df_prophet.iloc[split_index:]

# Defining and Training Prophet Model
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False  # Not useful if data < 1 year
)
'''
Initializing a Prophet model and configuring how it should treat seasonality.
  Parameter	                          Meaning
• daily_seasonality=True     ---   Learn patterns that repeat every day (e.g., higher consumption in evenings)
• weekly_seasonality=True    ---   Capture variations across days of the week (e.g., weekends vs weekdays)
• yearly_seasonality=False   ---   Skip yearly patterns — since dataset covers less than 1 year

'''

# Fit the model
model.fit(train_prophet)

# Creating future timestamps equal to the length of test set
future = model.make_future_dataframe(periods=len(test_prophet), freq='H')
'''
model.make_future_dataframe() creates a new DataFrame with:
The same time index as the training data, plus. Additional future time points (the forecast horizon)
  Parameter	    Meaning
• periods ---   How many future timestamps to generate (equal to the size of your test set)
• freq='H' ---  Frequency = hourly — because  data is hourly
'''

# Forecasting
forecast = model.predict(future)
'''
Generating a DataFrame called forecast with predicted values and uncertainty intervals for every timestamp in future, including both the training and test range
'''

# Getting predicted values for test period
forecast_test = forecast[['ds', 'yhat']].iloc[-len(test_prophet):].set_index('ds')
actual_test = test_prophet.set_index('ds')
'''
    Code Part	                     What It Does
•  forecast[['ds', 'yhat']] ---    Select only the date (ds) and forecast (yhat) columns
•  .iloc[-len(test_prophet):] ---  Take the last N rows, where N is the number of hours in test set.
•  .set_index('ds') ---           Set ds (datetime) as the index — same format as the actual test set.
'''

rmse_prophet = np.sqrt(mean_squared_error(actual_test['y'], forecast_test['yhat']))
'''
Setting the ds column (datetime) as the index for the actual test data, so it's aligned and can be compared directly with the forecast.
'''

print(f"Prophet RMSE: {rmse_prophet:.4f}")

'''
Output:
Prophet RMSE: 0.1147
'''

# Plotting forecast vs actual
fig = model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()


## Forecasting Model 3: LSTM (Long Short-Term Memory) ##

# Selecting features
features = ['consumption', 'temperature', 'humidity', 'wind_speed', 'avg_past_consumption']
data = df[features].copy()

# Normalizing the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=features)

# Creating Sequences for LSTM
def create_sequences(data, target_col, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, target_col])
    return np.array(X), np.array(y)
'''
Parameters :  
   Parameter	           What it is
• data ---         A NumPy array (scaled) of shape (n_samples, n_features)
• target_col ---	 The column index of the target feature (e.g., 0 for consumption)
• window_size ---	 How many previous time steps to use per prediction (e.g., 24 hours)

Loop : 
• Start at index 24, so you have enough past data to form a full sequence.
• Loop over the rest of the data — each loop iteration creates 1 sample for training or testing.

Inside Loop: 
• X.append(data[i-window_size:i]) - grabs the last window_size rows from the dataset
• Shape of each slice: (window_size, n_features)
• y.append(data[i, target_col]) - grabs the target value (e.g., consumption) at time i, right after the current window.

Output :
• X: shape (n_samples, 24, 5) — the past 24 hours of features for each sample
• y: shape (n_samples,) — the consumption value trying to predict at the next hour.
'''

# Choose window (how many past hours to look at)
window_size = 24

# Column index of target ('consumption' is first in list, so index 0)
target_col_index = 0

X, y = create_sequences(scaled_data, target_col=target_col_index, window_size=window_size)
'''
Use the past 24 hours to predict the consumption (column 0)
'''

# Train-Test split (80%)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Building and Training LSTM model
model = Sequential()
'''
Sequential() - creates a Keras sequential model, where you can stack layers one after another in order. 
'''

model.add(LSTM(units=64, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
'''
Argument	Explanation
units=64 --- Number of memory units (also called "neurons") in the LSTM layer. More units → more learning capacity.
activation='tanh' --- Activation function. tanh is commonly used in LSTM for smooth gradient flow.
return_sequences=False --- Tells the LSTM to output only the last output in the sequence. Since we're doing a single prediction, not a sequence, this should be False.
input_shape=(X_train.shape[1], X_train.shape[2]) --- Defines the shape of the input data: (timesteps, features). For example: (24, 5).

'''

model.add(Dense(1))
'''
Dense(1) means a fully connected output layer with 1 neuron — this neuron outputs the predicted value: electricity consumption for the next hour. As this is a regression problem, so just a single continuous output.
'''

model.compile(optimizer='adam', loss='mse')
'''
Parameter	           Purpose
optimizer='adam' --- Adam optimizer is an adaptive learning algorithm, good for time series
loss='mse' ---       Mean Squared Error — used because this is a regression task

'''

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
'''
Parameter	Explanation
X_train, y_train --- Training data (input sequences and target values)
epochs=10 --- Number of times the model sees the entire training set
batch_size=32 --- Number of samples processed before model weights are updated
validation_split=0.1 --- 10% of training data used to evaluate model performance during training
verbose=1 --- Shows training progress (loss/val_loss) in output
'''

# Predicting and Evaluating
y_pred = model.predict(X_test)

# Inverse scale (only for consumption)
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), scaled_data.shape[1]-1))], axis=1))[:, 0]
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((len(y_pred), scaled_data.shape[1]-1))], axis=1))[:, 0]

# RMSE
rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"LSTM RMSE: {rmse_lstm:.4f}")

'''
Output:
LSTM RMSE: 0.1148
'''

# Visualize Plot

plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted (LSTM)', alpha=0.8)
plt.title('LSTM Forecast vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.show()


## Forecasting Model 4: XGBoost (Extreme Gradient Boosting) ## 

# Dropping missing values from lag/rolling featuresdf_xgb = df.dropna()
df_xgb = df.dropna()

# Target variable
y = df_xgb['consumption']

# Feature columns
features = [
    'temperature', 'humidity', 'wind_speed', 'avg_past_consumption',
    'lag_1', 'lag_24', 'rolling_mean_24',
    'hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday'
]

X = df_xgb[features]

# Train/Test Split
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Training the XGBoost Model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
'''
Parameter	              Description
n_estimators=100 --- The number of trees (boosting rounds) to build. More trees → higher model complexity and potentially better accuracy (but risk of overfitting).
learning_rate=0.1 --- Controls the contribution of each tree. Lower = slower but more precise learning. Called "eta" in XGBoost. Common values: 0.01 – 0.3
max_depth=5 --- The maximum depth of each decision tree. Deeper trees learn more complex patterns, but can overfit.
random_state=42 --- Sets a seed to make results reproducible across runs (important for consistency).
'''

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")

'''
Output:
XGBoost RMSE: 0.1119
'''

# Visualize Prediction 
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted (XGBoost)', alpha=0.8)
plt.title("XGBoost Forecast vs Actual")
plt.xlabel("Time Steps")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.show()




# === Final Comparison ===

results = {
    'Model': ['ARIMA', 'Prophet', 'LSTM', 'XGBoost'],
    'RMSE': [rmse_arima, rmse_prophet, rmse_lstm, rmse_xgb]
}

results_df = pd.DataFrame(results).sort_values(by='RMSE')

# Display sorted results
print("\n Final Model Comparison (sorted by RMSE):")
print(results_df)

'''
Output:
Final Model Comparison (sorted by RMSE):
     Model      RMSE
3  XGBoost  0.111932
0    ARIMA  0.114324
1  Prophet  0.114732
2     LSTM  0.114803

'''

# Conclusion #

'''
Rank	Model	   RMSE	        Interpretation
 1	XGBoost	   0.111932	Best performance — lowest error
 2	ARIMA	   0.114324	Performs well, especially for linear time series
 3	Prophet	   0.114732	Also good, captures calendar/holiday effects
 4	LSTM	   0.114803	Slightly higher error — needs more tuning/data
'''











