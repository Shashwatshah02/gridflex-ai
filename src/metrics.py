def compute_metrics(dispatch, assumptions=None):
    bill = (dispatch.grid_import_kwh*dispatch.buy_price_usd_kwh-dispatch.grid_export_kwh*dispatch.sell_price_usd_kwh).sum()
    return {'cost_usd': float(bill), 'grid_import_kwh': float(dispatch.grid_import_kwh.sum()),
            'grid_export_kwh': float(dispatch.grid_export_kwh.sum()),
            'estimated_emissions_kg': float((dispatch.grid_import_kwh*dispatch.emissions_kg_per_kwh).sum()),
            'battery_throughput_kwh': float((dispatch.charge_kwh+dispatch.discharge_kwh).sum())}
