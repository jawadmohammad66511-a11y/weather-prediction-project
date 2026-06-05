"""
Generates synthetic historical weather data (2009–2020) for 8 Indian cities,
mimicking the Kaggle dataset used in the base paper.
Saves to weather_india.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

CITIES = {
    "Bangalore":  {"base_tmax": 28, "base_tmin": 18, "humidity": 65, "pressure": 912,  "wind": 10, "rain_scale": 4},
    "Mumbai":     {"base_tmax": 32, "base_tmin": 24, "humidity": 75, "pressure": 1009, "wind": 14, "rain_scale": 8},
    "Delhi":      {"base_tmax": 30, "base_tmin": 15, "humidity": 50, "pressure": 1005, "wind": 9,  "rain_scale": 3},
    "Hyderabad":  {"base_tmax": 33, "base_tmin": 20, "humidity": 55, "pressure": 947,  "wind": 11, "rain_scale": 4},
    "Chennai":    {"base_tmax": 34, "base_tmin": 23, "humidity": 72, "pressure": 1010, "wind": 15, "rain_scale": 6},
    "Kolkata":    {"base_tmax": 31, "base_tmin": 20, "humidity": 70, "pressure": 1008, "wind": 10, "rain_scale": 6},
    "Pune":       {"base_tmax": 30, "base_tmin": 18, "humidity": 55, "pressure": 943,  "wind": 8,  "rain_scale": 5},
    "Ahmedabad":  {"base_tmax": 35, "base_tmin": 20, "humidity": 45, "pressure": 1008, "wind": 10, "rain_scale": 2},
}

dates = pd.date_range("2009-01-01", "2020-12-31", freq="D")
rows = []

for city, p in CITIES.items():
    for date in dates:
        doy    = date.dayofyear
        year   = date.year
        month  = date.month

        # Seasonal sine wave
        season = np.sin(2 * np.pi * (doy - 80) / 365)  # peak ~summer

        # Monthly modifiers (monsoon = months 6–9)
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        tmax = (p["base_tmax"]
                + 7 * season
                + np.random.normal(0, 1.5)
                + 0.02 * (year - 2009))         # slight warming trend

        tmin = (p["base_tmin"]
                + 5 * season
                + np.random.normal(0, 1.2)
                + 0.015 * (year - 2009))

        humidity = (p["humidity"]
                    + 20 * is_monsoon
                    - 10 * (season * 0.5)
                    + np.random.normal(0, 5))
        humidity = np.clip(humidity, 10, 100)

        # Precipitation: near zero most days, spikes in monsoon
        if is_monsoon and np.random.rand() < 0.45:
            precip = np.random.exponential(p["rain_scale"] * 3)
        elif np.random.rand() < 0.08:
            precip = np.random.exponential(p["rain_scale"])
        else:
            precip = 0.0

        pressure = (p["pressure"]
                    + np.random.normal(0, 3)
                    - 2 * is_monsoon)

        wind = (p["wind"]
                + 5 * is_monsoon
                + np.random.normal(0, 2))
        wind = max(wind, 0)

        rows.append({
            "city":        city,
            "date":        date,
            "year":        year,
            "month":       month,
            "day":         date.day,
            "day_of_year": doy,
            "tmax":        round(tmax, 2),
            "tmin":        round(tmin, 2),
            "humidity":    round(humidity, 1),
            "precipitation": round(max(precip, 0), 2),
            "pressure":    round(pressure, 1),
            "wind_speed":  round(wind, 1),
        })

df = pd.DataFrame(rows)
df.to_csv(df.to_csv("weather_india.csv", index=False))
print(f"Dataset created: {len(df):,} rows × {len(df.columns)} cols")
print(df.head())
print(df.describe())