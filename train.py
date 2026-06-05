import pandas as pd
import numpy as np
import joblib, os, warnings, json
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_LEN   = 10
TARGETS   = ["tmax", "tmin", "humidity", "precipitation", "pressure", "wind_speed"]
FEATURES  = ["tmax", "tmin", "humidity", "precipitation", "pressure", "wind_speed",
             "month", "day_of_year"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

N_JOBS = -1   # -1 = use ALL CPU cores; set to 4 if you want to limit

# ── Load & clean data ──────────────────────────────────────────────────────────
CSV_PATH = "weather_india.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌  '{CSV_PATH}' not found. Place it next to train.py.")

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df["city"]        = df["city"].str.strip().str.title()
df                = df.sort_values(["city", "date"]).reset_index(drop=True)
df["month"]       = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

before = len(df)
df = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df)}")

CITIES = df["city"].unique().tolist()
print(f"Cities ({len(CITIES)}): {CITIES}")
print(f"Training all cities IN PARALLEL using {os.cpu_count()} CPU cores...\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_sequences(city_df, target, scaler):
    vals = scaler.transform(city_df[FEATURES])
    X, y = [], []
    for i in range(SEQ_LEN, len(vals)):
        X.append(vals[i - SEQ_LEN:i].flatten())
        y.append(city_df[target].iloc[i])
    return np.array(X), np.array(y)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def get_models():
    return {
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200, max_depth=12, min_samples_leaf=2,
                                 n_jobs=1, random_state=42),   # n_jobs=1 inside parallel
        "XGBoost":           XGBRegressor(
                                 n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=0, n_jobs=1),
        "Linear Regression": LinearRegression(),
        "SVR":               SVR(kernel="rbf", C=10, epsilon=0.1),
    }

# ── Per-city training function (runs in parallel) ──────────────────────────────
def train_city(city, city_df):
    results = {}

    if len(city_df) < SEQ_LEN + 20:
        print(f"  ⚠️  [{city}] Not enough data ({len(city_df)} rows), skipping.")
        return city, {}

    raw_split = int(len(city_df) * 0.8)
    train_df  = city_df.iloc[:raw_split]
    test_df   = city_df.iloc[raw_split - SEQ_LEN:]

    for target in TARGETS:
        if city_df[target].isna().any():
            print(f"  ⚠️  [{city}] Skipping {target} — NaN values present.")
            continue

        scaler = MinMaxScaler()
        scaler.fit(train_df[FEATURES])

        X_train, y_train = make_sequences(train_df, target, scaler)
        X_test,  y_test  = make_sequences(test_df,  target, scaler)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠️  [{city}] Skipping {target} — insufficient sequences.")
            continue

        target_results = {}
        best_rmse_val  = np.inf
        best_model     = None
        best_name      = None

        for name, model in get_models().items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r     = rmse(y_test, preds)
                target_results[name] = round(r, 4)
                print(f"  [{city}] {target:15s} | {name:20s} | RMSE: {r:.4f}")

                if r < best_rmse_val:
                    best_rmse_val = r
                    best_model    = model
                    best_name     = name
            except Exception as e:
                print(f"  ⚠️  [{city}] {name} failed for {target}: {e}")

        if best_model is None:
            continue

        key = f"{city}_{target}"
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{key}_model.pkl"))
        joblib.dump(scaler,     os.path.join(MODEL_DIR, f"{key}_scaler.pkl"))

        results[target] = {
            "rmse_by_model": target_results,
            "best_model":    best_name,
            "best_rmse":     round(best_rmse_val, 4),
        }

    print(f"  ✅  [{city}] Done — {len(results)}/{len(TARGETS)} targets trained.")
    return city, results

# ── Run ALL cities in parallel ─────────────────────────────────────────────────
city_dfs = {city: df[df["city"] == city].reset_index(drop=True) for city in CITIES}

parallel_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
    delayed(train_city)(city, city_dfs[city]) for city in CITIES
)

all_results = {city: res for city, res in parallel_results if res}

# ── Save outputs ───────────────────────────────────────────────────────────────
PKL_MODELS = os.path.join(MODEL_DIR, "training_results.pkl")
PKL_ROOT   = "training_results.pkl"
JSON_PATH  = os.path.join(MODEL_DIR, "training_results.json")

joblib.dump(all_results, PKL_MODELS)
joblib.dump(all_results, PKL_ROOT)
with open(JSON_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅  Saved: {PKL_MODELS}")
print(f"✅  Saved: {PKL_ROOT}")
print(f"✅  Saved: {JSON_PATH}")

# ── RMSE Summary Table ─────────────────────────────────────────────────────────
print("\n\n📊  RMSE Comparison (avg across all targets per city):\n")
header = f"{'City':14s} | {'Rand Forest':>11} | {'XGBoost':>9} | {'Lin Reg':>9} | {'SVR':>9}"
print(header)
print("-" * len(header))

for city in CITIES:
    if city not in all_results:
        continue
    rfv, xgbv, lrv, svrv = [], [], [], []
    for target in TARGETS:
        if target not in all_results[city]:
            continue
        rd = all_results[city][target]["rmse_by_model"]
        if "Random Forest"     in rd: rfv.append(rd["Random Forest"])
        if "XGBoost"           in rd: xgbv.append(rd["XGBoost"])
        if "Linear Regression" in rd: lrv.append(rd["Linear Regression"])
        if "SVR"               in rd: svrv.append(rd["SVR"])

    rf_s  = f"{np.mean(rfv):.4f}"  if rfv  else "  N/A  "
    xgb_s = f"{np.mean(xgbv):.4f}" if xgbv else "  N/A  "
    lr_s  = f"{np.mean(lrv):.4f}"  if lrv  else "  N/A  "
    svr_s = f"{np.mean(svrv):.4f}" if svrv else "  N/A  "
    print(f"{city:14s} | {rf_s:>11} | {xgb_s:>9} | {lr_s:>9} | {svr_s:>9}")

print(f"\n\n🚀  All {len(CITIES)} cities trained in parallel. Done!")