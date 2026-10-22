"""Linear programming dispatch. SOC columns record interval-end energy."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from src.data import validate_hourly_data


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 20
    power_kw: float = 5
    efficiency: float = .95  # one-way efficiency; round trip is efficiency squared
    initial_fraction: float = .5
    min_fraction: float = .1
    max_fraction: float = .9
    dt_hours: float = 1
    throughput_penalty: float = 1e-5

    def validate(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError('Battery parameters must be finite.')
        if self.capacity_kwh < 0 or self.power_kw < 0 or not 0 < self.efficiency <= 1 or self.dt_hours != 1 or self.throughput_penalty <= 0:
            raise ValueError('Use nonnegative capacity/power, positive penalty, efficiency in (0,1], hourly intervals.')
        if not 0 <= self.min_fraction <= self.initial_fraction <= self.max_fraction <= 1:
            raise ValueError('Initial SOC must lie inside SOC limits.')


def assemble(day, charge, discharge, soc):
    out = day.copy().reset_index(drop=True)
    out['charge_kwh'], out['discharge_kwh'], out['soc_kwh'] = charge, discharge, soc
    net = out.load_kwh + out.charge_kwh - out.solar_kwh - out.discharge_kwh
    out['grid_import_kwh'] = np.maximum(net, 0)
    out['grid_export_kwh'] = np.maximum(-net, 0)
    return out


def validate_dispatch(out, battery, terminal=True, tol=1e-6):
    battery.validate()
    validate_hourly_data(out)
    c, d, s = (out[k].to_numpy() for k in ['charge_kwh', 'discharge_kwh', 'soc_kwh'])
    initial = battery.initial_fraction * battery.capacity_kwh
    previous = np.r_[initial, s[:-1]]
    balance = out.solar_kwh + out.discharge_kwh + out.grid_import_kwh - out.load_kwh - out.charge_kwh - out.grid_export_kwh
    if not np.isfinite(out[['charge_kwh','discharge_kwh','soc_kwh','grid_import_kwh','grid_export_kwh']]).all().all():
        raise ValueError('Nonfinite dispatch.')
    checks = [np.max(np.abs(balance)), np.max(np.abs(s - previous - battery.efficiency*c + d/battery.efficiency)),
              np.max(-c), np.max(-d), np.max(c - battery.power_kw*battery.dt_hours), np.max(d - battery.power_kw*battery.dt_hours),
              np.max(battery.min_fraction*battery.capacity_kwh-s), np.max(s-battery.max_fraction*battery.capacity_kwh),
              -out.grid_import_kwh.min(), -out.grid_export_kwh.min()]
    if terminal:
        checks.append(abs(s[-1]-initial))
    if max(checks) > tol:
        raise ValueError(f'Dispatch violates physical constraints: {max(checks):.3g}')
    return {'max_balance_residual_kwh': float(np.max(np.abs(balance))),
            'simultaneous_battery_intervals': int(((c > tol) & (d > tol)).sum()),
            'simultaneous_grid_intervals': int(((out.grid_import_kwh > tol) & (out.grid_export_kwh > tol)).sum())}


def optimize_dispatch(day, battery=BatteryConfig()):
    validate_hourly_data(day)
    battery.validate()
    n = len(day)
    # blocks: charge, discharge, import, export, end SOC
    objective = np.r_[np.full(2*n, battery.throughput_penalty), day.buy_price_usd_kwh, -day.sell_price_usd_kwh, np.zeros(n)]
    a = np.zeros((2*n+1, 5*n)); b = np.zeros(2*n+1)
    initial = battery.capacity_kwh*battery.initial_fraction
    for t in range(n):
        a[t, [t,n+t,2*n+t,3*n+t]] = [-1,1,1,-1]
        b[t] = day.load_kwh.iloc[t]-day.solar_kwh.iloc[t]
        a[n+t,[t,n+t,4*n+t]] = [-battery.efficiency,1/battery.efficiency,1]
        if t:
            a[n+t,4*n+t-1] = -1
        else:
            b[n+t] = initial
    a[-1,-1] = 1; b[-1] = initial
    bounds = [(0,battery.power_kw*battery.dt_hours)]*(2*n)+[(0,None)]*(2*n)+[(battery.min_fraction*battery.capacity_kwh,battery.max_fraction*battery.capacity_kwh)]*n
    result = linprog(objective, A_eq=a, b_eq=b, bounds=bounds, method='highs')
    if not result.success:
        raise ValueError(f'Battery solver failed ({result.status}): {result.message}')
    out = assemble(day, result.x[:n], result.x[n:2*n], result.x[4*n:])
    validate_dispatch(out, battery)
    return out
