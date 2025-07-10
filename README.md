# Project Overview
To predict short-term (next few hours/days) or long-term (weeks/months) electricity demand using machine learning and time series forecasting techniques such as ARIMA, LSTM, Prophet and XGBoost.

## 1. Dataset Overview
### Source: 
Kaggle - Smart Meter Electricity Consumption</br>
### Description: 
It contains hourly energy consumption, temperature, and timestamps.</br>
### Column Summary:
- datetime: Time of the reading (hourly or more frequent).</br>
- consumption: Energy used in kWh for that period.</br>
- temperature: Ambient temperature during the reading.</br>
- humidity, wind_speed: Extra weather variables that affect energy usage.</br>
- mean_consumption_last_24h / std_consumption_last_7d: Rolling consumption statistics indicating recent usage trends.</br>
- is_anomaly: Boolean flag for anomalous readings (spikes or drops).

### Business Value of Each Column:
- datetime: Enables precise forecasting for planning and operations.</br>
- consumption: The key metric for understanding how much energy will be needed and when.</br>
- Weather features: Provide context like Demand spikes during heatwaves or cold spells, guiding purchase and grid planning.</br>
- Historical stats: Capture short-term trends and help auto-adjust forecasts based on recent usage behavior.</br>
- Anomaly labels: Ensure forecasts are robust, ignoring spurious data to prevent bad buying or load-balancing decisions.

  



