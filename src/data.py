"""Synthetic simulation data; timestamps are UTC interval starts."""
import numpy as np
import pandas as pd

REQUIRED = ['load_kwh', 'solar_kwh', 'buy_price_usd_kwh', 'sell_price_usd_kwh', 'emissions_kg_per_kwh']


def generate_data(seed=42, n_days=120):
    if n_days < 10:
        raise ValueError('Generate at least 10 days.')
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2026-01-01', periods=n_days * 24, freq='h', tz='UTC')
    h = ts.hour.to_numpy()
    cloud = np.repeat(rng.uniform(.05, .85, n_days), 24)
    daylight = np.maximum(0, np.sin(np.pi * (h - 6) / 12))
    solar = 9 * daylight * (1 - .7 * cloud) * rng.uniform(.9, 1.1, len(ts))
    load = np.maximum(.1, 2.3 + 3.5 * np.exp(-((h - 19) / 3) ** 2) + .7 * (ts.day_of_week < 5) + rng.normal(0, .25, len(ts)))
    df = pd.DataFrame({'timestamp': ts, 'load_kwh': load, 'solar_kwh': solar,
                       'cloud_cover': cloud, 'buy_price_usd_kwh': np.where((h >= 16) & (h < 21), .38, .14),
                       'sell_price_usd_kwh': .04, 'emissions_kg_per_kwh': np.where((h >= 16) & (h < 21), .48, .32)})
    validate_hourly_data(df)
    return df


def validate_hourly_data(df):
    if len(df) == 0 or not {'timestamp', *REQUIRED}.issubset(df.columns):
        raise ValueError('Missing rows or required columns.')
    ts = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing or not ts.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Timestamps must be unique, sorted, continuous hourly intervals.')
    values = df[REQUIRED].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Energy, tariffs and emissions must be finite and nonnegative.')
    if (df.sell_price_usd_kwh > df.buy_price_usd_kwh).any():
        raise ValueError('Export credit must not exceed import price.')
