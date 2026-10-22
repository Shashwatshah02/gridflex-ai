"""Optional browser QA: pip install playwright; playwright install chromium.

Start Streamlit separately. Browser tooling is not a runtime dependency.
"""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    Path('docs').mkdir(exist_ok=True)
    with sync_playwright() as p:
        executable = os.environ.get('GRIDFLEX_CHROMIUM_PATH')
        browser = p.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        errors = []
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8501')
        page.get_by_text('Tomorrow’s energy, hour by hour').wait_for(timeout=60000)
        page.locator('.js-plotly-plot').first.wait_for()
        page.screenshot(path='docs/dashboard.png',full_page=True)
        page.get_by_role('tab',name='Battery & grid').click()
        page.get_by_role('tab',name='Results & assumptions').click()
        page.get_by_text('Download hourly comparison CSV').wait_for()
        page.screenshot(path='docs/results.png',full_page=True)
        assert not errors, errors
        print('Browser QA passed: charts, tabs, CSV control; screenshots saved.')
        browser.close()


if __name__=='__main__':
    main()
