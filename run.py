##### DISCLAIMER PLEASE READ! // BITTE LESEN! ######

# Use at your own risk! / Benutzen auf eigene Gefahr!
# DO NOT use this code to inform real investment decisions.
# Verwenden Sie diesen Code NICHT, um echte Anlageentscheidungen zu treffen.
# No financial advice is given or implied.
# Es wird keine Finanzberatung gegeben oder angedeutet.
# The author is NOT responsible for any losses incurred by using this code.
# Der Autor ist NICHT verantwortlich für Verluste, die durch die Verwendung dieses Codes entstehen.
# Parts of this code were generated with AI (GPT-5.2-Codex)
# Teile dieses Codes wurden mit KI (GPT-5.2-Codex) generiert

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

start_date = "2017-01-01"
end_date = "2026-05-07"
compare_ecb_rate = False # Invests taxes at the ECB Deposit Facility Rate
fixed_rate = 3 # If compare_ecb_rate is False, this fixed rate (in %) is used instead
end_date_ts = pd.Timestamp(end_date)

# AMUNDI ETF LEVERAGED MSCI USA DAILY UCITS ETF (EUR)
# ISIN FR0010755611   |  WKN A0X8ZS   |  Ticker 18MF 
# Source: https://my.oekb.at/kapitalmarkt-services/kms-output/fonds-info/sd/af/f?isin=FR0010755611
lev_AgEs = np.array([     # Datum, zu zahlende KESt, Einstandswertanpassung
  ["2017-12-14", 45.9023, 166.4050], 
  ["2018-10-23", 51.8614, 188.6954],
  ["2019-09-03", 37.0140, 142.2197],
  ["2020-09-16", 17.2034, 62.2565],
  #["2020-09-16", 17.2034, 62.2565], # Duplicate Entry
  ["2021-11-03", 216.0078, 785.7033],
  ["2022-12-20", 21.5122, 79.1534],
  #["2023-10-18", 66.1045, 240.1261], # Duplicate Entry when considering stock split
  ["2023-10-19", 0.2203, 0.8004], 
  ["2024-09-19", 1.2216, 4.4415],
  ["2026-01-28", 0.0, 0.0]
], dtype=object)

# Xtrackers MSCI USA UCITS ETF 1C (USD)
# ISIN IE00BJ0KDR00   |  WKN A1XB5V   |  Ticker XD9U 
# Source: https://my.oekb.at/kapitalmarkt-services/kms-output/fonds-info/sd/af/f?isin=IE00BJ0KDR00
unlev_AgEs = np.array([     # Datum, zu zahlende KESt, Einstandswertanpassung
  ["2017-07-31", 0.2970, 1.5136],
  ["2018-07-26", 0.4637, 2.0638],
  ["2019-07-09", 0.6498, 2.8441],
  ["2020-07-06", 1.0581, 4.3726],
  ["2021-04-29", 1.4998, 5.9763],
  ["2022-06-13", 1.6881, 6.6293],
  ["2023-07-12", 1.7504, 7.0234],
  ["2024-07-29", 0.9178, 3.9996],
  ["2025-07-30", 2.1257, 8.4628]
], dtype=object)

def apply_split_adjustment(ages, rows, factor):
  ages[:rows, 1] = ages[:rows, 1].astype(float) * factor
  ages[:rows, 2] = ages[:rows, 2].astype(float) * factor

def convert_ages_to_eur(ages, exchange):
  for i, date in enumerate(ages[:, 0]):
    fx = exchange.loc[date]
    ages[i][1] *= fx
    ages[i][2] *= fx

def ensure_ages_dates_present(ages_list, index):
  for date in np.concatenate([ages[:, 0] for ages in ages_list]):
    if date not in index:
      print(f"Error: Date {date} from AgE data is missing in share prices!")
      exit(1)

def advance_series(series, current_date, pct):
  series.append(series[-1].copy())
  series[-1][0] = current_date
  change = pct.loc[current_date]
  if isinstance(change, pd.Series):
    change = change.iloc[0]
  series[-1][1] *= (1 + float(change))

def apply_ages_for_date(series, ages, current_date):
  date_str = current_date.strftime("%Y-%m-%d")
  if date_str not in ages[:, 0]:
    return
  num_shares = series[-1][2]
  tax = ages[ages[:, 0] == date_str][0][1] * num_shares
  einstand = ages[ages[:, 0] == date_str][0][2] * num_shares
  val_prev = series[-1][1]
  series[-1][1] -= tax
  series[-1][2] *= (val_prev - tax) / val_prev  # "Sell" shares to pay the tax
  series[-1][3] += einstand
  series[-1][4] += tax

def accrue_tax_cash(series, current_date, rate_series):
  rate = rate_series.loc[current_date]
  if pd.isna(rate):
    return
  series[-1][4] *= (1 + float(rate) / 100.0 / 360.0)

def apply_end_tax(series, rate, days=1):
  series.append(series[-1].copy())
  val_prev = series[-1][1]
  taxes = val_prev - series[-1][3]  # Current Value - Einstandswert
  taxes = max(0, taxes * rate)
  series[-1][0] += pd.Timedelta(days=days)
  series[-1][1] -= taxes
  series[-1][2] = 0
  series[-1][3] = 0
  series[-1][4] += taxes

def extend_series(series, days):
  series.append(series[-1].copy())
  series[-1][0] += pd.Timedelta(days=days)

def get_tax_rate_series(compare_ecb_rate, fixed_rate, ecb_series, index):
  if compare_ecb_rate:
    return ecb_series
  return pd.Series(fixed_rate, index=index, dtype=float)

apply_split_adjustment(lev_AgEs, rows=6, factor=1 / 300)

def yf_download_cache(ticker, start, end, cache_path):
  if pd.io.common.file_exists(cache_path):
    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    if not cached.empty and cached.index.max().date() == pd.Timestamp.utcnow().date():
      if "Close" in cached.columns:
        return cached["Close"].loc[:end]
      return cached.iloc[:, 0].loc[:end]
  fresh = yf.download(ticker, start=start, end=end)
  close = fresh["Close"] if isinstance(fresh, pd.DataFrame) else fresh
  close = close.loc[:end]
  (close.to_frame("Close") if isinstance(close, pd.Series) else close).to_csv(cache_path)
  return close
  
exchange = yf_download_cache("USDEUR=X", start_date, end_date, "cache_USDEUR=X.csv")
lev_raw = yf_download_cache("18MF.DE", start_date, end_date, "cache_18MF.DE.csv")
unlev_raw = yf_download_cache("XD9U.DE", start_date, end_date, "cache_XD9U.DE.csv")

ecb_df = pd.read_csv("ecb_deposit_facility.csv", parse_dates=["DATE"])
ecb_series = (
  ecb_df[["DATE", "Deposit facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.DFR.LEV)"]]
  .rename(columns={
    "Deposit facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.DFR.LEV)": "Level"
  })
  .set_index("DATE")["Level"]
  .astype(float)
)
ecb_series = ecb_series.loc[:end_date_ts]

common_index = exchange.index.intersection(lev_raw.index).intersection(unlev_raw.index).intersection(ecb_series.index)
common_index = common_index[common_index <= end_date_ts]
exchange = exchange.reindex(common_index)
lev_raw = lev_raw.reindex(common_index)
unlev_raw = unlev_raw.reindex(common_index)
ecb_series = ecb_series.reindex(common_index).ffill()
tax_rate_series = get_tax_rate_series(compare_ecb_rate, fixed_rate, ecb_series, common_index)

convert_ages_to_eur(unlev_AgEs, exchange)
ensure_ages_dates_present([lev_AgEs, unlev_AgEs], common_index)
lev_pct = lev_raw.pct_change()
unlev_pct = unlev_raw.pct_change()

lev_after_tax = []
lev_fic = [] # Fictional values with tax payments only at the end of the period
unlev_after_tax = []
unlev_fic = [] # Fictional values with tax payments only at the end of the period
it = 0



# Calculate Running Values After Tax, Adjusting for AgEs

for current_date in common_index:
  if it == 0:
    lev_val = lev_raw.loc[current_date]
    unlev_val = unlev_raw.loc[current_date]

    lev_after_tax.append([current_date, lev_val, 1., lev_val, 0.])    # Date, Value, Shares, Einstandswert, Tax cash (ECB DFR)
    unlev_after_tax.append([current_date, unlev_val, 1., unlev_val, 0.])  # Date, Value, Shares, Einstandswert, Tax cash (ECB DFR)

    lev_fic.append([current_date, lev_val, 1., lev_val, 0.])    # Date, Value, Shares, Einstandswert, Tax cash (ECB DFR)
    unlev_fic.append([current_date, unlev_val, 1., unlev_val, 0.])  # Date, Value, Shares, Einstandswert, Tax cash (ECB DFR)
  else:
    advance_series(lev_after_tax, current_date, lev_pct)
    advance_series(unlev_after_tax, current_date, unlev_pct)
    advance_series(lev_fic, current_date, lev_pct)
    advance_series(unlev_fic, current_date, unlev_pct)

  it+=1

  accrue_tax_cash(lev_after_tax, current_date, tax_rate_series)
  accrue_tax_cash(unlev_after_tax, current_date, tax_rate_series)
  accrue_tax_cash(lev_fic, current_date, tax_rate_series)
  accrue_tax_cash(unlev_fic, current_date, tax_rate_series)

  apply_ages_for_date(lev_after_tax, lev_AgEs, current_date)
  apply_ages_for_date(unlev_after_tax, unlev_AgEs, current_date)


# Pay Taxes at the End of the Period

apply_end_tax(lev_after_tax, rate=0.275)
apply_end_tax(unlev_after_tax, rate=0.275)
apply_end_tax(lev_fic, rate=0.275)
apply_end_tax(unlev_fic, rate=0.275)


# Repeat for 6 months to make the final point more visible in the plot
extend_series(lev_after_tax, days=180)
extend_series(unlev_after_tax, days=180)
extend_series(lev_fic, days=180)
extend_series(unlev_fic, days=180)



# Plotting

lev_after_tax = np.array(lev_after_tax)
unlev_after_tax = np.array(unlev_after_tax)
lev_fic = np.array(lev_fic)
unlev_fic = np.array(unlev_fic)

# Normalize to 100 at the start date
lev_val_start = lev_after_tax[0, 1]
lev_after_tax[:, 1] = 100 * lev_after_tax[:, 1] / lev_val_start
lev_after_tax[:, 3:5] = 100 * lev_after_tax[:, 3:5] / lev_val_start

unlev_val_start = unlev_after_tax[0, 1]
unlev_after_tax[:, 1] = 100 * unlev_after_tax[:, 1] / unlev_val_start
unlev_after_tax[:, 3:5] = 100 * unlev_after_tax[:, 3:5] / unlev_val_start

lev_fic[:, 1] = 100 * lev_fic[:, 1] / lev_val_start
lev_fic[:, 3:5] = 100 * lev_fic[:, 3:5] / lev_val_start

unlev_fic[:, 1] = 100 * unlev_fic[:, 1] / unlev_val_start
unlev_fic[:, 3:5] = 100 * unlev_fic[:, 3:5] / unlev_val_start

series = [
  (lev_after_tax, "2x Leveraged (A0X8ZS) with AgEs", "red"),
  (unlev_after_tax, "Unleveraged (A1XB5V) with AgEs", "blue"),
  (lev_fic, "2x Leveraged (A0X8ZS) (all tax at end of period)", "orange"),
  (unlev_fic, "Unleveraged (A1XB5V) (all tax at end of period)", "green"),
]

metrics = [
  (1, "Value of holdings (normalized to 100)"),
  (2, "Number of Shares Held"),
  (3, "Gesetzlicher Einstandswert (normalized to 100)"),
  (4, "Tax (invested at ECB Deposit Facility Rate) (normalized to 100)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for ax, (col, title) in zip(axes.flat, metrics):
  for data, label, color in series:
    dates = pd.to_datetime(data[:, 0])
    ax.plot(dates, data[:, col].astype(float), label=label, color=color)
  ax.set_title(title)
  ax.set_xlabel("Date")
  if metrics[0][0] == col or metrics[2][0] == col or metrics[3][0] == col:
    ax.set_ylabel("Normalized value")
  elif metrics[1][0] == col:
    ax.set_ylabel("Shares")
  ax.grid(True, alpha=0.3)
  ax.legend()

for ax in axes[0, :]:
  ax.tick_params(axis="x", labelbottom=True)

plt.tight_layout()
plt.savefig("comparison_plot_1.png", dpi=300)
#plt.show()

metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for ax, (data, label, _) in zip(axes2.flat, series):
  dates = pd.to_datetime(data[:, 0])
  ax2 = ax.twinx()
  for (col, metric_title), color in zip(metrics, metric_colors):
    if col == 2:
      ax2.plot(dates, data[:, col].astype(float), label=metric_title, color=color)
    else:
      ax.plot(dates, data[:, col].astype(float), label=metric_title, color=color)
  ax.set_title(label)
  ax.set_xlabel("Date")
  ax.set_ylabel("Normalized value")
  ax2.set_ylabel("Shares")
  ax.grid(True, alpha=0.3)
  handles, labels = ax.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  legend = ax.legend(handles + handles2, labels + labels2, loc=(0.05, 0.65), fontsize="small")
  legend.set_zorder(10)

for ax in axes2[0, :]:
  ax.tick_params(axis="x", labelbottom=True)

plt.tight_layout()
plt.savefig("comparison_plot_2.png", dpi=300)
#plt.show()


def format_final(name, data):
  value = float(data[-1, 1])
  shares = float(data[-1, 2])
  cost_basis = float(data[-1, 3])
  paid_tax = float(data[-1, 4])
  return (
    f"{name}: value={value:.2f}, paid_tax={paid_tax:.2f}  "
  )

def calc_annualized_metrics(data, trading_days=252):
  dates = pd.to_datetime(data[:, 0])
  values = pd.Series(data[:, 1].astype(float), index=dates).sort_index()
  start = values.iloc[0]
  end = values.iloc[-1]
  days = (values.index[-1] - values.index[0]).days
  cagr = (end / start) ** (365.0 / days) - 1.0 if days > 0 else float("nan")
  daily_returns = values.pct_change().dropna()
  vol = daily_returns.std() * (trading_days ** 0.5)
  return cagr, vol

print("Final values (normalized):")
print(format_final("2x Leveraged (A0X8ZS) with AgEs", lev_after_tax))
print(format_final("Unleveraged (A1XB5V) with AgEs", unlev_after_tax))
print(format_final("2x Leveraged (A0X8ZS) end-tax", lev_fic))
print(format_final("Unleveraged (A1XB5V) end-tax", unlev_fic))

print("\nAnnualized return / volatility:")
for name, series_data in [
  ("2x Leveraged (A0X8ZS) with AgEs", lev_after_tax),
  ("Unleveraged (A1XB5V) with AgEs", unlev_after_tax),
  ("2x Leveraged (A0X8ZS) end-tax", lev_fic),
  ("Unleveraged (A1XB5V) end-tax", unlev_fic),
]:
  cagr, vol = calc_annualized_metrics(series_data)
  print(f"{name}: CAGR={cagr:.2%}, vol={vol:.2%}")