# ----------------------------------------------------------------------------
# Data Extraction.py
# Introduction to Credit Risk and Applications in Python Seminar
# Authors: Sharma & Miriashtiani
# Description: Extracts business bankruptcy counts from AOUSC F-2 PDFs,
#              merges with quarterly establishment counts from QCEW CSVs,
#              computes default rates (PD) and logit(PD), and
#              augments with lagged FRED macro series.
# ----------------------------------------------------------------------------

import re
import glob
import pathlib
import textwrap
import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
from fredapi import Fred
from pandas.tseries.offsets import QuarterEnd

###############################################################################
# 1. Utilities
###############################################################################

# Directory containing raw data files
RAW_DIR    = pathlib.Path('raw')          # folder with PDFs & CSV
PDF_GLOB   = str(RAW_DIR / 'f2_*.pdf')   # pattern for AOUSC Table-F-2 files


def business_ch7_ch11_from_total_line(pdf_path: str) -> tuple[int, int]:
    """
    Return (bus_ch7, bus_ch11) from an AOUSC Table F-2 PDF.
    Works for all quarters since 2007.

    Layout logic (verified on 2016-2025 PDFs):
      • 'Total' row has 14 numbers.
      • Indices 6 and 7 are Business Chapter 7 and Chapter 11 counts.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or '').splitlines():
                # Look for the 'Total' or 'TOTAL' row
                if re.match(r'^(Total|TOTAL)\b', line.strip()):
                    # Extract numbers with optional commas
                    nums = re.findall(r'\d{1,3}(?:,\d{3})*', line)
                    if len(nums) >= 8:
                        # Parse and return business Ch-7 and Ch-11
                        bus_ch7  = int(nums[6].replace(',', ''))
                        bus_ch11 = int(nums[7].replace(',', ''))
                        return bus_ch7, bus_ch11
    raise ValueError(f"'Total' line not found in {pdf_path}")


def quarter_from_fname(fname: str) -> str:
    """
    Extract 'YYYYqX' quarter code from filename 'f2_YYYYqX.pdf'.
    """
    return re.search(r'f2_(\d{4}q[1-4])', fname).group(1)

###############################################################################
# 2. Build the defaults DataFrame
###############################################################################

records = []
for pdf in glob.glob(PDF_GLOB):
    # Derive quarter and extract bankruptcy counts
    quarter          = quarter_from_fname(pdf)
    bus_ch7, bus_ch11 = business_ch7_ch11_from_total_line(pdf)
    records.append({
        "quarter"  : quarter,
        "bus_ch7"  : bus_ch7,
        "bus_ch11" : bus_ch11,
        "bus_total": bus_ch7 + bus_ch11      # sum of Ch-7 & Ch-11
    })

# Create DataFrame indexed by quarter
defaults = (pd.DataFrame(records)
              .set_index("quarter")
              .sort_index())

print("\n=== Business Ch-7 & Ch-11 counts ===")
print(defaults.head())

###############################################################################
# 3. Build exposure DataFrame
###############################################################################

EXPO_GLOB = 'raw/expo_*.csv'          # path pattern for QCEW CSVs

def total_estabs(csv_path):
    """
    Return national quarterly establishment count from QCEW CSV.
    Filters for agglvl_code == 10 (nationwide), own_code == 0, industry_code == 10.
    """
    df = pd.read_csv(
        csv_path,
        usecols=['agglvl_code', 'own_code', 'industry_code', 'qtrly_estabs']
    )

    # Select national total row
    nat = df.query(
        "(agglvl_code == 10) & (own_code == 0) & (industry_code == 10)"
    )
    if nat.empty:
        raise ValueError(f"No national-total row found in {csv_path!r}")

    # Convert string with commas to integer
    val = (
        nat['qtrly_estabs']
          .astype(str)
          .str.replace(',', '', regex=False)
          .astype(int)
          .iloc[0]
    )
    return val

records = []
for csv in glob.glob(EXPO_GLOB):
    # Extract quarter code and count establishments
    qtr = re.search(r'expo_(\d{4}q[1-4])', csv).group(1)
    records.append({'quarter': qtr,
                    'establishment_count': total_estabs(csv)})

expo = (pd.DataFrame(records)
          .set_index('quarter')
          .sort_index())

print("\n=== Built exposure DataFrame ===")
print(expo.head())

# Quick plot of exposure over time
expo.plot(title="Total establishments (all industries)")
plt.show()

###############################################################################
# 4. Compute PD and logit(PD)
###############################################################################

panel = defaults.join(expo, how='inner')

# Choose numerator: bus_ch11 or bus_total
NUMERATOR = panel["bus_total"]

# Compute default probability and its logit
panel["pd"]       = NUMERATOR / panel["establishment_count"]
panel["logit_pd"] = np.log(panel["pd"] / (1 - panel["pd"]))

###############################################################################
# 5. Pull macro series from FRED and lag (fredapi version)
###############################################################################

fred = Fred(api_key='99853075d21a4d65ce632a14d26edac0')  # replace with your key

FRED_IDS = {
    'GDPC1'       : 'gdp_level',
    'TB3MS'       : 't3m_rate',
    'BAMLH0A0HYM2': 'hy_oas',
    'UNRATE'      : 'unemp_rate',
    'DCOILWTICO'  : 'wti_price',
    'DTWEXBGS'    : 'usd_twex'
}
START = '2014-10-01'

def fred_series(series_id, start=START):
    """
    Fetches a FRED series as pandas Series starting from 'start'.
    """
    s = fred.get_series(series_id, observation_start=start)
    return s.rename(series_id).astype(float)

# Pull and concatenate all series
macro = pd.concat(
    {nick: fred_series(fid) for fid, nick in FRED_IDS.items()},
    axis=1
)

# Convert to quarterly frequency by mean and compute GDP QoQ growth
macro_q = macro.resample('Q-DEC').mean()
macro_q['gdp_qoq_pct'] = macro_q['gdp_level'].pct_change() * 100
macro_q = macro_q.drop(columns=['gdp_level'])

# Lag series by one quarter
macro_lag1 = macro_q.shift(1, freq=QuarterEnd())

print("\n=== Lagged macro DataFrame (head) ===")
print(macro_lag1.head())

# Align panel index to timestamp quarter ends and merge
panel.index = (
    panel.index
          .str.replace(r'(\d{4})q([1-4])', r'\1Q\2', regex=True)
          .astype('period[Q-DEC]')
          .to_timestamp('Q')
)
panel = panel.join(macro_lag1, how='inner')
# Save final panel
panel.to_csv('panel_data.csv', index=True)
