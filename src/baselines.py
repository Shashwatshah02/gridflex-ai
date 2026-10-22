import numpy as np
from src.data import validate_hourly_data
from src.optimize import BatteryConfig, assemble, validate_dispatch


def no_battery_dispatch(day):
    validate_hourly_data(day)
    return assemble(day, np.zeros(len(day)), np.zeros(len(day)), np.zeros(len(day)))


def rule_based_dispatch(day, battery=BatteryConfig()):
    """Solar-surplus charging with a terminal SOC reachability reserve.

    Replenish from the grid only when necessary to finish at initial SOC.
    This explicit terminal correction makes daily bill comparisons fair.
    """
    validate_hourly_data(day); battery.validate()
    initial = s = battery.capacity_kwh*battery.initial_fraction
    low, high = battery.capacity_kwh*battery.min_fraction, battery.capacity_kwh*battery.max_fraction
    p, eta = battery.power_kw*battery.dt_hours, battery.efficiency
    cs, ds, ss = [], [], []
    for t, row in enumerate(day.itertuples()):
        remaining = len(day)-t-1
        floor = max(low, initial-remaining*p*eta)
        ceiling = min(high, initial+remaining*p/eta)
        surplus = row.solar_kwh-row.load_kwh
        c = min(max(surplus,0),p,max(0,(ceiling-s)/eta))
        d = min(max(-surplus,0),p,max(0,(s-floor)*eta))
        next_s = s+eta*c-d/eta
        if next_s < floor:
            c = (floor-s)/eta; d = 0
        elif next_s > ceiling:
            d = (s-ceiling)*eta; c = 0
        s += eta*c-d/eta
        cs.append(c); ds.append(d); ss.append(s)
    out = assemble(day,cs,ds,ss)
    validate_dispatch(out,battery)
    return out
