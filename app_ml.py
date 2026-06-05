"""
app_ml.py  –  Flask backend for ML-based weather prediction
Combines:
  • Real-time current weather via OpenWeatherMap (for live display)
  • Next-day predictions via trained Random Forest / XGBoost models
  • RMSE comparison table for all models per city
  • 15-day rolling forecast via iterative ML prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, joblib, numpy as np, os, warnings
warnings.filterwarnings("ignore")

app  = Flask(__name__)
CORS(app)

API_KEY   = "d6bec2e9fb2c7d7128e2cfb718f59e56"
MODEL_DIR = "models"

TARGETS  = ["tmax", "tmin", "humidity", "precipitation", "pressure", "wind_speed"]
FEATURES = ["tmax", "tmin", "humidity", "precipitation", "pressure", "wind_speed",
            "month", "day_of_year"]
SEQ_LEN  = 10

# Model registry
_model_cache  = {}
_scaler_cache = {}

SUPPORTED_CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "Ahmedabad"
]

def load_model(city, target):
    key = f"{city}_{target}"
    if key not in _model_cache:
        mp = os.path.join(MODEL_DIR, f"{key}_model.pkl")
        sp = os.path.join(MODEL_DIR, f"{key}_scaler.pkl")
        if not os.path.exists(mp):
            return None, None
        _model_cache[key]  = joblib.load(mp)
        _scaler_cache[key] = joblib.load(sp)
    return _model_cache[key], _scaler_cache[key]

_training_results = None
def get_training_results():
    global _training_results
    if _training_results is None:
        p = os.path.join(MODEL_DIR, "training_results.pkl")
        _training_results = joblib.load(p) if os.path.exists(p) else {}
    return _training_results

def build_feature_row(weather_row, month, doy):
    return np.array([
        weather_row.get("tmax",          weather_row.get("temp_max", 30)),
        weather_row.get("tmin",          weather_row.get("temp_min", 20)),
        weather_row.get("humidity",      60),
        weather_row.get("precipitation", 0),
        weather_row.get("pressure",      1010),
        weather_row.get("wind_speed",    10),
        month,
        doy,
    ], dtype=float)

def owm_to_weather_row(d):
    return {
        "tmax":          d["main"]["temp_max"],
        "tmin":          d["main"]["temp_min"],
        "humidity":      d["main"]["humidity"],
        "precipitation": d.get("rain", {}).get("1h", 0),
        "pressure":      d["main"]["pressure"],
        "wind_speed":    round(d["wind"]["speed"] * 3.6, 1),
    }


@app.route("/")
def home():
    return "ML Weather Prediction API Running"


@app.route("/weather", methods=["POST"])
def get_weather():
    try:
        data = request.get_json()
        city = data.get("city", "")
        url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        d    = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": d.get("message","API error")}), 400
        return jsonify({
            "city":        d["name"],
            "temperature": d["main"]["temp"],
            "feels_like":  d["main"]["feels_like"],
            "humidity":    d["main"]["humidity"],
            "wind_speed":  round(d["wind"]["speed"] * 3.6, 1),
            "visibility":  d.get("visibility", 0) / 1000,
            "temp_min":    d["main"]["temp_min"],
            "temp_max":    d["main"]["temp_max"],
            "description": d["weather"][0]["description"],
            "condition":   d["weather"][0]["main"],
            "weather_id":  d["weather"][0]["id"],
            "country":     d["sys"]["country"],
            "lat":         d["coord"]["lat"],
            "lon":         d["coord"]["lon"],
            "pressure":    d["main"]["pressure"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        city       = data.get("city", "Mumbai")
        model_name = data.get("model", "Random Forest")

        city_key = next((c for c in SUPPORTED_CITIES if c.lower() == city.lower()), None)
        if city_key is None:
            return jsonify({"error": f"City '{city}' not supported. Choose from: {SUPPORTED_CITIES}"}), 400

        url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return jsonify({"error": "OWM API error"}), 400
        live = owm_to_weather_row(resp.json())

        import datetime
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        month    = tomorrow.month
        doy      = tomorrow.timetuple().tm_yday

        rng         = np.random.default_rng(42)
        predictions = {}
        model_used  = {}

        for target in TARGETS:
            model, scaler = load_model(city_key, target)
            if model is None:
                predictions[target] = None
                continue

            seq_rows = []
            for lag in range(SEQ_LEN, 0, -1):
                noise_scale = 0.03 * lag
                row = build_feature_row({
                    "tmax":          live["tmax"]          + rng.normal(0, noise_scale * live["tmax"]),
                    "tmin":          live["tmin"]          + rng.normal(0, noise_scale * live["tmin"]),
                    "humidity":      live["humidity"]      + rng.normal(0, noise_scale * 10),
                    "precipitation": max(0, live["precipitation"] + rng.normal(0, 0.5)),
                    "pressure":      live["pressure"]      + rng.normal(0, 1),
                    "wind_speed":    max(0, live["wind_speed"]  + rng.normal(0, 1)),
                }, month, doy)
                seq_rows.append(row)

            seq_rows.append(build_feature_row(live, month, doy))
            seq_arr  = np.array(seq_rows[-SEQ_LEN:])
            seq_norm = scaler.transform(seq_arr)
            X        = seq_norm.flatten().reshape(1, -1)

            pred = float(model.predict(X)[0])
            pred = max(0, pred) if target == "precipitation" else pred
            predictions[target] = round(pred, 2)
            model_used[target]  = type(model).__name__

        return jsonify({
            "city":        city_key,
            "current":     live,
            "predictions": predictions,
            "model_used":  model_used,
            "note":        "Next-day predictions from trained ML models"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ml_forecast", methods=["POST"])
def ml_forecast():
    try:
        data     = request.get_json()
        city     = data.get("city", "Hyderabad")
        city_key = next((c for c in SUPPORTED_CITIES if c.lower() == city.lower()), None)
        if city_key is None:
            return jsonify({"error": f"City '{city}' not in ML cities: {SUPPORTED_CITIES}"}), 400

        url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return jsonify({"error": "OWM error"}), 400
        live = owm_to_weather_row(resp.json())

        import datetime
        today = datetime.date.today()

        rng     = np.random.default_rng(0)
        history = []
        for lag in range(SEQ_LEN):
            m   = (today - datetime.timedelta(days=SEQ_LEN - lag)).month
            doy = (today - datetime.timedelta(days=SEQ_LEN - lag)).timetuple().tm_yday
            history.append(build_feature_row(live, m, doy))

        forecast_days = []
        for d in range(1, 16):
            future_date = today + datetime.timedelta(days=d)
            month       = future_date.month
            doy         = future_date.timetuple().tm_yday

            day_preds = {"date": str(future_date), "month": month, "doy": doy}
            new_row   = {}

            for target in TARGETS:
                model, scaler = load_model(city_key, target)
                if model is None:
                    day_preds[target] = None
                    continue
                seq_arr  = np.array(history[-SEQ_LEN:])
                seq_norm = scaler.transform(seq_arr)
                X        = seq_norm.flatten().reshape(1, -1)
                pred     = float(model.predict(X)[0])
                pred     = max(0, pred) if target == "precipitation" else pred
                day_preds[target] = round(pred, 2)
                new_row[target]   = pred

            new_row.setdefault("tmax",          day_preds.get("tmax",          live["tmax"]))
            new_row.setdefault("tmin",          day_preds.get("tmin",          live["tmin"]))
            new_row.setdefault("humidity",      day_preds.get("humidity",      live["humidity"]))
            new_row.setdefault("precipitation", day_preds.get("precipitation", 0))
            new_row.setdefault("pressure",      day_preds.get("pressure",      live["pressure"]))
            new_row.setdefault("wind_speed",    day_preds.get("wind_speed",    live["wind_speed"]))
            history.append(build_feature_row(new_row, month, doy))

            forecast_days.append(day_preds)

        return jsonify({"city": city_key, "forecast": forecast_days})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rmse_comparison", methods=["GET"])
def rmse_comparison():
    try:
        results = get_training_results()
        if not results:
            return jsonify({"error": "Training results not found. Run train_models.py first."}), 404

        summary  = []
        detailed = []

        for city, city_data in results.items():
            rf_vals, xgb_vals, lr_vals = [], [], []
            detail_row = {"city": city}

            for target in TARGETS:
                td = city_data.get(target, {})

                # Handle both pkl formats:
                # Format A (nested): {"rmse_by_model": {"Random Forest": 1.2, ...}}
                # Format B (flat):   {"Random Forest": 1.2, ...}
                rmse_map = td.get("rmse_by_model", td)

                rf  = rmse_map.get("Random Forest")
                xgb = rmse_map.get("XGBoost")
                lr  = rmse_map.get("Linear Regression")

                if rf  is not None: rf_vals.append(rf)
                if xgb is not None: xgb_vals.append(xgb)
                if lr  is not None: lr_vals.append(lr)

                # Keys match exactly what frontend JS expects:
                # rfKey  = `${target}_Random_Forest`
                # xgbKey = `${target}_XGBoost`
                # lrKey  = `${target}_Linear_Regression`
                detail_row[f"{target}_Random_Forest"]     = round(rf,  3) if rf  is not None else None
                detail_row[f"{target}_XGBoost"]           = round(xgb, 3) if xgb is not None else None
                detail_row[f"{target}_Linear_Regression"] = round(lr,  3) if lr  is not None else None

            detailed.append(detail_row)
            summary.append({
                "city":              city,
                "Random_Forest":     round(np.mean(rf_vals),  3) if rf_vals  else None,
                "XGBoost":           round(np.mean(xgb_vals), 3) if xgb_vals else None,
                "Linear_Regression": round(np.mean(lr_vals),  3) if lr_vals  else None,
            })

        return jsonify({"summary": summary, "detailed": detailed})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ml_cities", methods=["GET"])
def ml_cities():
    return jsonify({"cities": SUPPORTED_CITIES})


INDIAN_CITIES = [
    {"city": "Hyderabad",  "state": "Telangana",      "district": "Hyderabad"},
    {"city": "Mumbai",     "state": "Maharashtra",    "district": "Mumbai"},
    {"city": "Delhi",      "state": "Delhi",          "district": "New Delhi"},
    {"city": "Bangalore",  "state": "Karnataka",      "district": "Bengaluru Urban"},
    {"city": "Chennai",    "state": "Tamil Nadu",     "district": "Chennai"},
    {"city": "Kolkata",    "state": "West Bengal",    "district": "Kolkata"},
    {"city": "Jaipur",     "state": "Rajasthan",      "district": "Jaipur"},
    {"city": "Ahmedabad",  "state": "Gujarat",        "district": "Ahmedabad"},
    {"city": "Pune",       "state": "Maharashtra",    "district": "Pune"},
    {"city": "Lucknow",    "state": "Uttar Pradesh",  "district": "Lucknow"},
]

@app.route("/all_states", methods=["GET"])
def all_states():
    try:
        results = []
        for entry in INDIAN_CITIES:
            try:
                url  = f"http://api.openweathermap.org/data/2.5/weather?q={entry['city']}&appid={API_KEY}&units=metric"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    d = resp.json()
                    results.append({
                        "state":       entry["state"],
                        "district":    entry["district"],
                        "temperature": round(d["main"]["temp"], 1),
                        "condition":   d["weather"][0]["main"],
                        "humidity":    d["main"]["humidity"],
                        "wind_speed":  round(d["wind"]["speed"] * 3.6, 1),
                        "description": d["weather"][0]["description"],
                        "weather_id":  d["weather"][0]["id"],
                    })
            except Exception:
                pass
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        city = data.get("city", "")
        url  = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        fd   = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": fd.get("message", "API error")}), 400
        fl = []
        for i in range(0, min(40, len(fd["list"])), 8):
            dd = fd["list"][i]
            fl.append({
                "date":        dd["dt_txt"],
                "dt":          dd["dt"],
                "temp":        dd["main"]["temp"],
                "description": dd["weather"][0]["description"],
                "condition":   dd["weather"][0]["main"],
                "weather_id":  dd["weather"][0]["id"],
                "humidity":    dd["main"]["humidity"],
                "wind_speed":  round(dd["wind"]["speed"] * 3.6, 1),
            })
        return jsonify({"forecast": fl})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/weather_coords", methods=["POST"])
def weather_coords():
    try:
        data     = request.get_json()
        lat, lon = data["lat"], data["lon"]
        url  = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        d    = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": d.get("message", "API error")}), 400
        return jsonify({
            "city":        d["name"],
            "country":     d["sys"]["country"],
            "temperature": d["main"]["temp"],
            "feels_like":  d["main"]["feels_like"],
            "humidity":    d["main"]["humidity"],
            "wind_speed":  round(d["wind"]["speed"] * 3.6, 1),
            "visibility":  d.get("visibility", 0) / 1000,
            "temp_min":    d["main"]["temp_min"],
            "temp_max":    d["main"]["temp_max"],
            "description": d["weather"][0]["description"],
            "condition":   d["weather"][0]["main"],
            "weather_id":  d["weather"][0]["id"],
            "lat":         d["coord"]["lat"],
            "lon":         d["coord"]["lon"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast_coords", methods=["POST"])
def forecast_coords():
    try:
        data     = request.get_json()
        lat, lon = data["lat"], data["lon"]
        url  = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=6)
        fd   = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": fd.get("message", "API error")}), 400
        fl = []
        for i in range(0, min(40, len(fd["list"])), 8):
            dd = fd["list"][i]
            fl.append({
                "date":        dd["dt_txt"],
                "dt":          dd["dt"],
                "temp":        dd["main"]["temp"],
                "description": dd["weather"][0]["description"],
                "condition":   dd["weather"][0]["main"],
                "weather_id":  dd["weather"][0]["id"],
                "humidity":    dd["main"]["humidity"],
                "wind_speed":  round(dd["wind"]["speed"] * 3.6, 1),
            })
        return jsonify({"forecast": fl})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)