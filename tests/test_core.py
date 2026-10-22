import numpy as np
import pandas as pd
import pytest
from src.data import generate_data, validate_hourly_data
from src.optimize import BatteryConfig, optimize_dispatch, validate_dispatch
from src.baselines import rule_based_dispatch, no_battery_dispatch
from src.metrics import compute_metrics
from src.forecast import train_forecasters, forecast_day
from src.evaluate import compare_day


@pytest.fixture
def day():
    return generate_data(n_days=10).iloc[:24].copy()


def test_reproducible_data():
    pd.testing.assert_frame_equal(generate_data(),generate_data())
    df = generate_data()
    assert df.loc[(df.timestamp.dt.hour<=6)|(df.timestamp.dt.hour>=18),'solar_kwh'].max()<1e-10


@pytest.mark.parametrize('mutation',['negative','nan','duplicate','gap','tariff'])
def test_invalid_data(day,mutation):
    if mutation=='negative': day.loc[0,'load_kwh']=-1
    if mutation=='nan': day.loc[0,'solar_kwh']=np.nan
    if mutation=='duplicate': day.loc[1,'timestamp']=day.loc[0,'timestamp']
    if mutation=='gap': day=day.drop(index=1)
    if mutation=='tariff': day.loc[0,'sell_price_usd_kwh']=1
    with pytest.raises(ValueError): validate_hourly_data(day)


@pytest.mark.parametrize('method',[optimize_dispatch,rule_based_dispatch])
@pytest.mark.parametrize('scenario',['normal','zero_solar','zero_battery','zero_power','full','empty','flat','zero_prices','large_solar'])
def test_dispatch_constraints(day,method,scenario):
    config = BatteryConfig()
    if scenario=='zero_solar': day['solar_kwh']=0
    if scenario=='zero_battery': config=BatteryConfig(capacity_kwh=0)
    if scenario=='zero_power': config=BatteryConfig(power_kw=0)
    if scenario=='full': config=BatteryConfig(initial_fraction=.9)
    if scenario=='empty': config=BatteryConfig(initial_fraction=.1)
    if scenario=='flat': day['buy_price_usd_kwh']=.2
    if scenario=='zero_prices': day[['buy_price_usd_kwh','sell_price_usd_kwh']]=0
    if scenario=='large_solar': day['solar_kwh']*=10
    result = method(day,config)
    checks = validate_dispatch(result,config)
    assert checks['simultaneous_battery_intervals']==0
    assert checks['simultaneous_grid_intervals']==0
    if scenario=='zero_battery':
        assert compute_metrics(result)['cost_usd']==pytest.approx(compute_metrics(no_battery_dispatch(day))['cost_usd'])
    if scenario=='zero_solar':
        assert result.discharge_kwh.sum() <= result.charge_kwh.sum()*config.efficiency**2+1e-7


def test_hand_calculated_arbitrage(day):
    day['solar_kwh']=0; day['load_kwh']=2
    day['buy_price_usd_kwh']=.1; day.loc[12:,'buy_price_usd_kwh']=.5
    battery = BatteryConfig(capacity_kwh=4,power_kw=2,efficiency=1,initial_fraction=0,min_fraction=0,max_fraction=1)
    out=optimize_dispatch(day,battery)
    assert compute_metrics(out)['cost_usd']==pytest.approx(12.8)
    assert out.charge_kwh.sum()==pytest.approx(4)


def test_metrics_manual(day):
    day['solar_kwh']=0; day['load_kwh']=1; day['buy_price_usd_kwh']=.2; day['emissions_kg_per_kwh']=.3
    result=compute_metrics(no_battery_dispatch(day))
    assert result['cost_usd']==pytest.approx(4.8)
    assert result['estimated_emissions_kg']==pytest.approx(7.2)


def test_no_future_observation_leakage():
    df=generate_data(n_days=10)
    model=train_forecasters(df.iloc[:192])
    future=df.iloc[192:216].copy()
    first=forecast_day(model,df.iloc[:192],future)
    future[['load_kwh','solar_kwh','cloud_cover']]=999
    second=forecast_day(model,df.iloc[:192],future)
    np.testing.assert_array_equal(first[['load_kwh','solar_kwh']],second[['load_kwh','solar_kwh']])
    with pytest.raises(ValueError): forecast_day(model,df.iloc[:216],future)


def test_actual_replay(day):
    forecast=day.copy(); forecast['load_kwh']*=2
    schedules,_=compare_day(day,forecast,BatteryConfig())
    plan=optimize_dispatch(forecast,BatteryConfig())
    np.testing.assert_allclose(schedules['Optimized'].charge_kwh,plan.charge_kwh)
    validate_dispatch(schedules['Optimized'],BatteryConfig())


def test_invalid_config(day):
    for config in [BatteryConfig(efficiency=0),BatteryConfig(capacity_kwh=-1),BatteryConfig(dt_hours=.5),BatteryConfig(initial_fraction=1)]:
        with pytest.raises(ValueError): optimize_dispatch(day,config)
