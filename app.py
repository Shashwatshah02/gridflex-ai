"""Run with streamlit run app.py."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data import generate_data
from src.forecast import evaluate_forecasters, forecast_day
from src.optimize import BatteryConfig, validate_dispatch
from src.evaluate import compare_day

st.set_page_config(page_title='GridFlex AI', page_icon='☀️', layout='wide')
st.markdown('''<style>.stApp{background:#0b1421;color:#edf5f4} [data-testid="stMetric"]{background:#142636;padding:18px;border-radius:12px} h1{letter-spacing:-1.5px}</style>''',unsafe_allow_html=True)
st.caption('GRIDFLEX AI  /  ENERGY PLANNING LAB')
st.title('Make the most of every solar hour.')
st.write('Explore day-ahead battery schedules, electricity costs and grid demand.')
st.info('Synthetic simulation data · UTC hourly intervals · Decision support only')

@st.cache_data
def setup():
    df = generate_data()
    models, forecast, errors, split = evaluate_forecasters(df)
    return df,models,errors,split

try:
    df,models,errors,split = setup()
    with st.sidebar:
        st.header('Scenario studio')
        dates = df.iloc[split:].timestamp.dt.date.unique()
        date = st.selectbox('Planning date (UTC)',dates)
        capacity = st.slider('Battery capacity (kWh)',0,80,20)
        power = st.slider('Maximum power (kW)',0,20,5)
        efficiency = st.slider('One-way efficiency',.70,1.0,.95,.01)
        initial = st.slider('Starting SOC (%)',10,90,50)
        solar = st.slider('Solar scale',0.0,3.0,1.0,.1)
        load = st.slider('Demand scale',.1,3.0,1.0,.1)
        tariff = st.selectbox('Tariff',['Time of use','Flat'])
        st.caption('Round-trip efficiency = one-way efficiency². Grid charging and export are allowed.')
    actual = df.loc[df.timestamp.dt.date==date].copy().reset_index(drop=True)
    history = df.loc[df.timestamp < actual.timestamp.iloc[0]]
    forecast = forecast_day(models,history,actual)
    for frame in [actual,forecast]:
        frame['solar_kwh'] *= solar; frame['load_kwh'] *= load
        if tariff=='Flat':
            frame['buy_price_usd_kwh'] = .20
    battery = BatteryConfig(capacity_kwh=capacity,power_kw=power,efficiency=efficiency,initial_fraction=initial/100)
    schedules,metrics = compare_day(actual,forecast,battery)
    optimized = schedules['Optimized']; kpi = metrics.loc['Optimized']
    cols = st.columns(5)
    for col,label,value in zip(cols,['Optimized bill','No battery bill','Cost savings','Grid import','Est. emissions'],[f'${kpi.cost_usd:.2f}',f'${metrics.loc["No battery","cost_usd"]:.2f}',f'{kpi.savings_pct:.1f}%',f'{kpi.grid_import_kwh:.1f} kWh',f'{kpi.estimated_emissions_kg:.1f} kg']):
        col.metric(label,value)
    st.caption('Bills and grid flows replay forecast battery schedules against actual synthetic energy. Forecast errors are balanced by unlimited grid import/export.')
    overview,dispatch,details = st.tabs(['Energy outlook','Battery & grid','Results & assumptions'])
    with overview:
        st.subheader('Tomorrow’s energy, hour by hour')
        fig = go.Figure()
        for label,frame,column,color,dash in [('Forecast demand',forecast,'load_kwh','#64d8ff','solid'),('Forecast solar',forecast,'solar_kwh','#ffc76b','solid'),('Actual demand',actual,'load_kwh','#64d8ff','dot'),('Actual solar',actual,'solar_kwh','#ffc76b','dot')]:
            fig.add_scatter(x=frame.timestamp,y=frame[column],name=label,line=dict(color=color,dash=dash))
        fig.update_layout(yaxis_title='Energy (kWh / hourly interval)',template='plotly_dark')
        st.plotly_chart(fig,width='stretch')
        st.subheader('Chronological holdout forecast performance')
        st.dataframe(errors,hide_index=True,width='stretch')
    with dispatch:
        fig = go.Figure()
        fig.add_bar(x=optimized.timestamp,y=optimized.charge_kwh,name='Charge (kWh)',marker_color='#60e1ae')
        fig.add_bar(x=optimized.timestamp,y=-optimized.discharge_kwh,name='Discharge (kWh)',marker_color='#ffbc73')
        fig.add_scatter(x=optimized.timestamp,y=optimized.soc_kwh,name='End SOC (kWh)',line_color='#89abff')
        fig.update_layout(template='plotly_dark',yaxis_title='kWh',barmode='relative')
        st.plotly_chart(fig,width='stretch')
        fig = go.Figure()
        for name,frame in schedules.items():
            fig.add_scatter(x=frame.timestamp,y=frame.grid_import_kwh,name=name)
        fig.update_layout(template='plotly_dark',yaxis_title='Grid imports (kWh)')
        st.plotly_chart(fig,width='stretch')
        tariff_fig = go.Figure()
        for column in ['buy_price_usd_kwh','sell_price_usd_kwh']:
            tariff_fig.add_scatter(x=actual.timestamp,y=actual[column],name=column)
        tariff_fig.update_layout(template='plotly_dark',yaxis_title='Tariff (USD/kWh)')
        st.plotly_chart(tariff_fig,width='stretch')
    with details:
        st.dataframe(metrics,width='stretch')
        combined = pd.concat([frame.assign(strategy=name) for name,frame in schedules.items()],ignore_index=True)
        st.download_button('Download hourly comparison CSV',combined.to_csv(index=False),'gridflex-dispatch.csv','text/csv')
        st.dataframe(combined,width='stretch')
        st.json(validate_dispatch(optimized,battery))
        st.markdown('''**Model assumptions**

- 120 synthetic days, seed 42; first 96 training days, final 24 holdout days. Fixed model; previous-day observations become available daily.
- Forecasts use calendar and previous-day energy; no actual future weather. Scenario scaling is applied after forecasting.
- One-hour intervals, 10–90% SOC bounds; ending SOC equals starting SOC for both battery strategies.
- Rule baseline charges on solar surplus, discharges against deficits, and reserves/replenishes enough energy for terminal SOC.
- LP minimizes import cost less export credit plus $0.00001/kWh throughput penalty. Displayed bills exclude the penalty.
- Pure LP does not enforce flow exclusivity; simultaneous flows are detected above. Grid import/export are unlimited; solar is exported without curtailment.
- Emissions use assumed hourly grid intensity (0.32 or 0.48 kg/kWh), excluding export displacement, embodied carbon and battery manufacture. Cost optimization can increase emissions.
- Synthetic performance is not field validation, investment advice or hardware control. No ROI or verified carbon savings are claimed.
''')
except ValueError as exc:
    st.error(str(exc))
