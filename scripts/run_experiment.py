"""Run from repository root: python -m scripts.run_experiment."""
import json
from pathlib import Path
from time import perf_counter
import pandas as pd
from src.data import generate_data
from src.forecast import evaluate_forecasters
from src.evaluate import compare_day
from src.optimize import BatteryConfig, validate_dispatch


def main():
    start = perf_counter()
    data = generate_data()
    _, forecast, errors, split = evaluate_forecasters(data)
    rows, hourly, checks = [], [], []
    for offset in range(0,len(forecast),24):
        actual = data.iloc[split+offset:split+offset+24].reset_index(drop=True)
        schedules, metrics = compare_day(actual,forecast.iloc[offset:offset+24].reset_index(drop=True),BatteryConfig())
        rows.append(metrics)
        for name,frame in schedules.items():
            hourly.append(frame.assign(strategy=name))
            if name!='No battery':
                checks.append(validate_dispatch(frame,BatteryConfig()))
    totals = pd.concat(rows).groupby(level=0).sum().drop(columns='savings_pct')
    totals['savings_pct'] = (totals.loc['No battery','cost_usd']-totals.cost_usd)/totals.loc['No battery','cost_usd']*100
    Path('reports').mkdir(exist_ok=True)
    data.to_csv('data/synthetic_hourly.csv',index=False)
    errors.to_csv('reports/forecast_metrics.csv',index=False)
    totals.to_csv('reports/strategy_comparison.csv')
    pd.concat(hourly).to_csv('reports/hourly_dispatch.csv',index=False)
    metadata = {'seed':42,'synthetic_days':120,'holdout_days':len(forecast)//24,'timezone':'UTC','runtime_seconds':perf_counter()-start,'max_balance_residual_kwh':max(c['max_balance_residual_kwh'] for c in checks),'simultaneous_battery_intervals':sum(c['simultaneous_battery_intervals'] for c in checks),'simultaneous_grid_intervals':sum(c['simultaneous_grid_intervals'] for c in checks)}
    Path('reports/experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(errors.to_string(index=False)); print(totals.to_string()); print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
