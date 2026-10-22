"""Forecast schedules replayed against realized energy with grid balancing."""
import pandas as pd
from src.baselines import no_battery_dispatch, rule_based_dispatch
from src.optimize import assemble, optimize_dispatch, validate_dispatch
from src.metrics import compute_metrics


def compare_day(actual, forecast, battery):
    schedules = {'No battery':no_battery_dispatch(actual)}
    for name,method in [('Rule based',rule_based_dispatch),('Optimized',optimize_dispatch)]:
        plan = method(forecast,battery)
        realized = assemble(actual,plan.charge_kwh.to_numpy(),plan.discharge_kwh.to_numpy(),plan.soc_kwh.to_numpy())
        validate_dispatch(realized,battery)
        schedules[name] = realized
    metrics = pd.DataFrame({name:compute_metrics(plan) for name,plan in schedules.items()}).T
    baseline = metrics.loc['No battery','cost_usd']
    metrics['savings_pct'] = (baseline-metrics.cost_usd)/baseline*100 if baseline>0 else float('nan')
    return schedules,metrics
