"""Day-ahead features use calendar and previous-day observations only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TARGETS = ['load_kwh', 'solar_kwh']


def features(day, previous):
    h = pd.to_datetime(day.timestamp).dt.hour.to_numpy()
    dow = pd.to_datetime(day.timestamp).dt.day_of_week.to_numpy()
    return np.column_stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), dow, *[previous[k].to_numpy() for k in TARGETS]])


def train_forecasters(train_df):
    if len(train_df) < 72 or len(train_df)%24:
        raise ValueError('Training requires at least three complete days.')
    x, y = [], []
    for start in range(24,len(train_df),24):
        day = train_df.iloc[start:start+24]
        x.append(features(day,train_df.iloc[start-24:start]))
        y.append(day[TARGETS].to_numpy())
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=4, random_state=42, n_jobs=1)
    model.fit(np.vstack(x),np.vstack(y))
    return {'model':model, 'training_end':train_df.timestamp.iloc[-1]}


def forecast_day(models, history_df, forecast_weather_df):
    """Third argument supplies future timestamps/tariffs, never actual weather."""
    day = forecast_weather_df.copy().reset_index(drop=True)
    if len(day)!=24 or len(history_df)<24 or history_df.timestamp.iloc[-1]>=day.timestamp.iloc[0]:
        raise ValueError('Need 24 future hours and strictly earlier history.')
    if models['training_end'] >= day.timestamp.iloc[0]:
        raise ValueError('Model training overlaps the forecast horizon.')
    prediction = np.maximum(0,models['model'].predict(features(day,history_df.iloc[-24:])))
    prediction[(pd.to_datetime(day.timestamp).dt.hour <= 6)|(pd.to_datetime(day.timestamp).dt.hour >= 18),1] = 0
    day[TARGETS] = prediction
    return day


def evaluate_forecasters(df):
    split = int(len(df)/24*.8)*24
    models = train_forecasters(df.iloc[:split])
    predictions, naive = [], []
    for start in range(split,len(df),24):
        predictions.append(forecast_day(models,df.iloc[:start],df.iloc[start:start+24]))
        naive.append(df.iloc[start-24:start][TARGETS].to_numpy())
    forecast = pd.concat(predictions,ignore_index=True)
    truth = df.iloc[split:][TARGETS].to_numpy()
    rows = []
    for name, values in [('random_forest',forecast[TARGETS].to_numpy()),('previous_day',np.vstack(naive))]:
        for j,target in enumerate(TARGETS):
            error = values[:,j]-truth[:,j]
            rows.append({'model':name,'target':target,'mae_kwh':float(np.abs(error).mean()),'rmse_kwh':float(np.sqrt((error**2).mean()))})
    return models, forecast, pd.DataFrame(rows), split
