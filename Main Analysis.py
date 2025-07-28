# ----------------------------------------------------------------------------
# Main Analysis.py
# Introduction to Credit Risk and Applications in Python Seminar
# Authors: Sharma & Miriashtiani
# Description: Runs AR(1) regression on logit PD, conducts OLS manually,
#              performs Monte-Carlo stress-testing with Student-t shocks,
#              and evaluates out-of-sample forecasts and key quantiles.
# ----------------------------------------------------------------------------

###############################################################################
# 0.  Imports
###############################################################################
import numpy as np                      # numerical operations and RNG
import pandas as pd                     # data handling
import scipy.stats as st                # statistical distributions
import matplotlib.pyplot as plt         # plotting
from matplotlib.pyplot import savefig   # saving figures
from scipy.linalg import inv            # matrix inversion

# Set random seed for reproducibility
np.random.seed(50)

# Constants and file paths
CSV      = "panel_data.csv"   # path to input CSV file
N_SIM    = 200_000             # number of Monte Carlo simulations
HORIZON  = 8                   # projection horizon (quarters)
DF_T     = 6                   # degrees of freedom for Student-t residuals

###############################################################################
# 1.  Load & prep data
###############################################################################
# Read panel data; first column becomes index (e.g., '2015q1')
df = pd.read_csv(CSV, index_col=0)

# Add AR(1) term: lagged logit PD, then drop initial NaN row
df["logit_pd_lag1"] = df["logit_pd"].shift(1)
df = df.dropna()

# Define target and regressors
TARGET  = "logit_pd"
AR_TERM = "logit_pd_lag1"
MACROS  = ['gdp_qoq_pct', 't3m_rate', 'hy_oas']

# Build response vector y and design matrix X
y = df[TARGET].to_numpy()  # shape (n, )
X = np.column_stack([
    np.ones(len(df)),            # constant term
    df[AR_TERM].to_numpy(),      # AR(1) lag
    df[MACROS].to_numpy()        # macro variables
])                             # shape (n, k)

# Store variable names and dimensions
var_names = ["const", AR_TERM] + MACROS
n, k = X.shape               # sample size and number of regressors

###############################################################################
# 2.  OLS by hand
###############################################################################
# Compute coefficients: beta = (X'X)^(-1) X'y
XtX_inv = inv(X.T @ X)
beta    = XtX_inv @ X.T @ y

# Calculate residuals and unbiased variance estimate
resid  = y - X @ beta
sigma2 = resid @ resid / (n - k)

# Variance-covariance, standard errors, t-stats, and two-tailed p-values
cov_beta = sigma2 * XtX_inv
se_beta  = np.sqrt(np.diag(cov_beta))
t_stats  = beta / se_beta
p_vals   = 2 * st.t.sf(np.abs(t_stats), df=n - k)

###############################################################################
# 3.  Print a regression-style table
###############################################################################
print("\n=== OLS results (statsmodels-free) ===")
print(f"Obs: {n:>4}    R²: {1 - resid.var()/y.var():.4f}")
header = f"{'Variable':<15}{'Coef':>12}{'StdErr':>12}{'t':>9}{'P>|t|':>9}"
print(header + "\n" + "-"*len(header))
for v, b, se, t, p in zip(var_names, beta, se_beta, t_stats, p_vals):
    print(f"{v:<15}{b:>12.4f}{se:>12.4f}{t:>9.2f}{p:>9.3f}")

###############################################################################
# 4.  Monte-Carlo stress-test engine (Student-t residuals)
###############################################################################
def simulate_paths(last_logit, last_macros, beta, sigma, n_sim=N_SIM,
                   horizon=HORIZON, df_t=DF_T, shock_dict=None):
    """
    Simulate PD paths under baseline or shocked conditions.

    Parameters:
      last_logit  – scalar logit(PD) at t = 0
      last_macros – array of current macro values (len = 3)
      beta        – OLS coefficients [const, phi, beta_macros...]
      sigma       – residual variance estimate
      shock_dict  – optional dict {quarter_offset: {macro: shock}}

    Returns:
      pd_paths – array of simulated PDs (n_sim × horizon)
    """
    # Unpack coefficients
    const, phi = beta[0], beta[1]
    beta_mac   = beta[2:]

    # Initialize macro paths as random walk baseline
    macros = np.tile(last_macros, (horizon, 1))
    if shock_dict:
        # Apply additive shocks in specified quarters
        for q, d in shock_dict.items():
            for name, val in d.items():
                j = MACROS.index(name)
                macros[q, j] += val

    # Generate Student-t residual shocks scaled by sigma
    eps = st.t.rvs(df=df_t, size=(n_sim, horizon)) * np.sqrt(sigma)

    # Initialize logit(PD) array
    logit = np.empty((n_sim, horizon))
    # First-quarter update
    logit[:, 0] = const + phi * last_logit + macros[0] @ beta_mac + eps[:, 0]
    # Propagate through horizon
    for t in range(1, horizon):
        logit[:, t] = (
            const + phi * logit[:, t-1] + macros[t] @ beta_mac + eps[:, t]
        )

    # Transform to PD
    pd_paths = 1 / (1 + np.exp(-logit))
    return pd_paths

# Generate baseline and scenario paths
last_row       = df.iloc[-1]
baseline_paths = simulate_paths(last_row[TARGET],
                                last_row[MACROS].to_numpy(),
                                beta, sigma2)
# Define combined shock scenario
combo = {
    "gdp_qoq_pct": -1.0,   # –1 pp GDP
    "t3m_rate"   : -0.30,  # –30 bp short rate
    "hy_oas"     : 0.50    # +50 bp HY-OAS
}
shock_dict = {0: combo.copy(), 1: combo.copy()}
scenario_paths = simulate_paths(last_row[TARGET],
                                last_row[MACROS].to_numpy(),
                                beta, sigma2,
                                shock_dict=shock_dict)

###############################################################################
# 5.  Plot baseline vs. scenario distribution
###############################################################################
plt.figure(figsize=(10, 6))

def plot_hist(paths, label, color, density=True):
    # Compute 1-year-ahead mean PD (%) and plot histogram
    pd_mean = paths[:, :4].mean(axis=1) * 100
    plt.hist(pd_mean, bins=80, alpha=0.5,
             label=label, color=color, density=density)
    # Mark percentiles
    for q, ls in zip([50, 95, 99], ['-', '--', ':']):
        v = np.percentile(pd_mean, q)
        plt.axvline(v, color=color, linestyle=ls, linewidth=1,
                    label=f"{label} {q}th = {v:.4f}%")

# Plot baseline and mild shock scenarios
plot_hist(baseline_paths, "Baseline", "royalblue", density=True)
plot_hist(scenario_paths, "Mild Shock", "darkorange", density=True)

plt.title("1-Year Ahead Mean PD – Baseline vs. Mild Shock")
plt.xlabel("PD (%)")
plt.ylabel("Density")
plt.xlim(0, 0.2)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
savefig("chart.png", dpi=300)
plt.show()

###############################################################################
# Sanity check & out-of-sample evaluation
###############################################################################
print("\nSign check (expect GDP −, short-rate +, HY-OAS +):")
for name, b in zip(["const", AR_TERM] + MACROS, beta):
    print(f"{name:12s}: {b:+.4f}")
# Check AR(1) coefficient
phi = beta[1]
print(f"\nAR(1) coefficient φ̂ = {phi:.3f}")

# Tail risk comparison
sigma     = np.sqrt(sigma2)
tail_gauss = st.norm.ppf(0.995) * sigma
tail_t     = st.t.ppf(0.995, df=DF_T) * sigma
print(f"\n99.5% Gaussian tail   : {tail_gauss:.4f}")
print(f"99.5% Student-t(df=6): {tail_t:.4f}")
print(f"fat-tail multiplier  : {tail_t / tail_gauss:.2f}")

# 4.1 Split data into train/test using date strings as indices
train = df.loc[:'2022-12-31'].copy()
test  = df.loc['2023-03-31':].copy()  # last 8 quarters

# 4.2 Fit OLS on training window
target_lag = train[TARGET].shift(1)
train["lag"] = target_lag
train = train.dropna()
Xtr = np.column_stack([
    np.ones(len(train)), train["lag"], train[MACROS]
])
ytr = train[TARGET]
beta_tr = inv(Xtr.T @ Xtr) @ Xtr.T @ ytr

# 4.3 Dynamic 1-step-ahead forecasts
pred = []
lag_val = train.iloc[-1][TARGET]
for _, row in test.iterrows():
    x_now = np.r_[1, lag_val, row[MACROS].values]
    y_hat = x_now @ beta_tr
    pred.append(y_hat)
    lag_val = y_hat  # use prediction for next step

# Attach predictions and compute PDs
test["logit_pred"] = pred
test["pd_pred"]    = 1 / (1 + np.exp(-test["logit_pred"]))
test["pd_actual"]  = 1 / (1 + np.exp(-test[TARGET]))

# 4.4 Compute out-of-sample MAPE
mape = (np.abs(test["pd_pred"] - test["pd_actual"]) / test["pd_actual"]).mean() * 100
print(f"\nOut-of-sample MAPE: {mape:.1f}%")

# 1-Year-Ahead quantiles for baseline vs. scenario
results_1y = []
for q in [50, 95, 99]:
    bq = np.percentile(baseline_paths[:, :4].mean(axis=1)*100, q)
    sq = np.percentile(scenario_paths[:, :4].mean(axis=1)*100, q)
    results_1y.append({
        "Quantile":f"{q}th", "Baseline": bq,
        "Scenario": sq, "Change": sq - bq
    })
df_1y = pd.DataFrame(results_1y)
print("\nKey 1-Year-Ahead PD Quantiles (%)")
print(df_1y.to_string(index=False, float_format="%.3f"))
