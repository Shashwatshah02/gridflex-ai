from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_default_and_disabled_battery():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'),default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric)==5
    app.sidebar.slider[0].set_value(0).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value==app.metric[1].value
    assert app.metric[2].value=='0.0%'
