from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import pyexcel as pe
import time

# Local file path
file_path = "file:///Users/jonathan/Astropy-stacking-with-AI/table.html"

# Start Chrome in headless mode
options = webdriver.ChromeOptions()
options.add_argument('--headless')
service = Service()  # optional, for explicit driver path
driver = webdriver.Chrome(options=options, service=service)

# Open the file
driver.get(file_path)

# Wait for JavaScript to populate the table
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "results"))
    )
except Exception as e:
    print("Table never appeared:", e)
    driver.quit()
    exit(1)

# Get HTML of the results table
table_html = driver.find_element(By.ID, "results").get_attribute('outerHTML')
driver.quit()

# Parse with pandas
df = pd.read_html(table_html)[0]
# Replace NaN with '' *only* in originally blank cells
df = df.where(pd.notna(df), '')

# Convert to ODS
records = df.values.tolist()
records.insert(0, df.columns.tolist())
pe.save_as(array=records, dest_file_name="output.ods")

print("Conversion to output.ods complete.")
