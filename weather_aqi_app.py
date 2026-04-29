"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    🌤️  Real-Time Weather + AQI Forecaster — Indian Cities Edition           ║
║    Sinusoidal ML  ·  Open-Meteo API  ·  Open-AQI  ·  Live Heatmaps         ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS APP DOES (in plain English):
──────────────────────────────────────
  This is a desktop weather dashboard for 20 Indian cities.
  It does three main things:

  1. FETCH real weather data from the internet (Open-Meteo API).
     We get the last 7 days of actual measured temperatures — not
     predictions from some other model, but raw sensor readings.

  2. TRAIN a simple math model on that data.
     The model assumes temperature follows a wave pattern every day
     (cool at night, warm at noon). It finds the best-fit wave and
     uses that to predict the next 24 hours.

  3. FETCH and PREDICT air quality (AQI).
     We pull real PM2.5/PM10/ozone data from Open-Meteo's air quality
     endpoint and fit a separate sinusoidal model to predict how
     pollution will behave in the next 24 hours.

WHY OPEN-METEO INSTEAD OF OPENWEATHERMAP?
──────────────────────────────────────────
  OpenWeatherMap's free tier gives you their own model's predictions.
  If we train our model on those predictions, we're essentially building
  a model that mimics another model — not real science, and the accuracy
  scores look terrible.

  Open-Meteo gives us:
    • past_days=7 → 7 days of REAL observed hourly temperatures
    • Air quality: PM2.5, PM10, O3, NO2 — also real observations
    • No API key → completely free, no registration needed

  Training on real measurements = much better accuracy scores (R²).

TABS IN THE APP:
────────────────
  📈 ML Forecast  — Temperature prediction using sinusoidal regression
  🌫️ AQI Monitor  — Air quality index + 24-hour pollution prediction
  🌡️ Heat Map     — Color-coded map of India showing temp/humidity/wind
  🏙️ City Compare — Bar chart comparing multiple cities side by side
  📊 Analytics    — 4-panel dashboard with humidity, wind, conditions, temps
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import threading
import math
from datetime import datetime, timedelta

# ── Third-Party: UI ───────────────────────────────────────────────────────────
import customtkinter as ctk

# ── Third-Party: Data & Networking ────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests

# ── Third-Party: ML / Scientific ─────────────────────────────────────────────
from scipy.optimize    import curve_fit          # fits our sine wave to data
from scipy.interpolate import griddata           # fills gaps between city dots on the map
from sklearn.metrics   import r2_score           # tells us how good the model is (0=bad, 1=perfect)

# ── Third-Party: Visualization ────────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Colors, Cities, API URLs
# ══════════════════════════════════════════════════════════════════════════════

# Open-Meteo weather endpoint (free, no API key needed)
OPENMETEO_URL     = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo air quality endpoint (same deal — free, no key)
OPENMETEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── App Color Palette (deep-space dark theme) ─────────────────────────────────
C = {
    "bg":      "#080c14",   # main background: almost black
    "card":    "#0f1623",   # card background: dark navy
    "card2":   "#161e2e",   # slightly lighter card
    "accent":  "#38bdf8",   # sky blue — primary highlight
    "accent2": "#818cf8",   # indigo — secondary highlight
    "accent3": "#fb923c",   # orange — warm/prediction color
    "accent4": "#34d399",   # emerald green — cool/good AQI
    "accent5": "#f472b6",   # pink — sine wave line
    "accent6": "#a78bfa",   # violet — AQI prediction
    "text":    "#e2e8f0",   # main text: light grey
    "dim":     "#64748b",   # dimmed text: medium grey
    "border":  "#1e293b",   # border lines: dark blue-grey
    "warning": "#fbbf24",   # yellow — moderate warning
    "error":   "#f87171",   # red — danger/bad AQI
    "aqi_good":      "#34d399",   # green
    "aqi_moderate":  "#fbbf24",   # yellow
    "aqi_sensitive": "#fb923c",   # orange
    "aqi_unhealthy": "#f87171",   # red
    "aqi_very":      "#c084fc",   # purple
    "aqi_hazardous": "#991b1b",   # dark red
}

# ── Indian Cities: name → (latitude, longitude) ──────────────────────────────
#   Latitude  = how far north/south (higher = more north)
#   Longitude = how far east/west  (higher = more east)
CITIES = {
    "Dehradun":    (30.3165,  78.0322),
    "Delhi":       (28.6139,  77.2090),
    "Mumbai":      (19.0760,  72.8777),
    "Bangalore":   (12.9716,  77.5946),
    "Chennai":     (13.0827,  80.2707),
    "Kolkata":     (22.5726,  88.3639),
    "Hyderabad":   (17.3850,  78.4867),
    "Pune":        (18.5204,  73.8567),
    "Jaipur":      (26.9124,  75.7873),
    "Lucknow":     (26.8467,  80.9462),
    "Chandigarh":  (30.7333,  76.7794),
    "Ahmedabad":   (23.0225,  72.5714),
    "Shimla":      (31.1048,  77.1734),
    "Manali":      (32.2396,  77.1887),
    "Rishikesh":   (30.0869,  78.2676),
    "Nainital":    (29.3919,  79.4542),
    "Varanasi":    (25.3176,  82.9739),
    "Patna":       (25.5941,  85.1376),
    "Bhubaneswar": (20.2961,  85.8245),
    "Guwahati":    (26.1445,  91.7362),
}

# ── AQI Breakpoints (US EPA standard — same used by India's CPCB) ─────────────
#   These thresholds convert raw PM2.5 concentration (µg/m³) into
#   the 0-500 AQI scale that's reported in the news.
#
#   Format: (pm25_low, pm25_high, aqi_low, aqi_high, label, color_key)
AQI_BREAKPOINTS = [
    (0.0,   12.0,   0,   50,  "Good",                  "aqi_good"),
    (12.1,  35.4,  51,  100,  "Moderate",              "aqi_moderate"),
    (35.5,  55.4, 101,  150,  "Unhealthy for Sensitive","aqi_sensitive"),
    (55.5, 150.4, 151,  200,  "Unhealthy",             "aqi_unhealthy"),
    (150.5,250.4, 201,  300,  "Very Unhealthy",        "aqi_very"),
    (250.5,500.4, 301,  500,  "Hazardous",             "aqi_hazardous"),
]

# ── AQI Health Advice — what each level means for a regular person ─────────────
AQI_ADVICE = {
    "Good":                   "🟢 Great air! Perfect for outdoor activities.",
    "Moderate":               "🟡 Acceptable. Very sensitive people may notice slight effects.",
    "Unhealthy for Sensitive":"🟠 Sensitive groups (asthma, elderly, kids) should limit outdoor time.",
    "Unhealthy":              "🔴 Everyone may start to experience health effects. Wear a mask outdoors.",
    "Very Unhealthy":         "🟣 Health alert! Avoid prolonged outdoor exertion.",
    "Hazardous":              "⚫ Emergency conditions. Stay indoors. Wear N95 if you must go out.",
}

# ── Simplified India border polygon (longitude, latitude pairs) ───────────────
#   Used to draw the outline of India on the heatmap.
#   Not perfectly accurate — just enough to show country boundaries.
_IND_LON = [
    68.2, 68.4, 69.3, 70.2, 71.1, 72.5, 72.8, 73.2, 73.3, 73.0,
    73.0, 73.8, 74.5, 75.1, 76.5, 77.5, 78.1, 79.0, 80.3, 80.3,
    80.1, 80.3, 80.5, 81.1, 82.0, 83.5, 85.1, 86.5, 87.5, 88.0,
    88.5, 89.5, 91.5, 92.5, 93.5, 94.6, 96.5, 97.4, 97.5, 97.1,
    96.5, 95.5, 94.5, 93.5, 92.1, 91.5, 90.5, 89.5, 88.5, 88.1,
    88.5, 88.1, 87.5, 86.5, 85.5, 84.5, 83.5, 82.5, 81.5, 80.5,
    79.5, 79.1, 78.5, 78.1, 77.5, 77.0, 76.5, 75.5, 74.5, 74.1,
    73.5, 73.0, 72.0, 71.5, 70.5, 70.0, 69.5, 69.0, 68.5, 68.2,
]
_IND_LAT = [
    22.5, 23.5, 24.0, 24.5, 24.8, 24.5, 23.0, 21.5, 20.0, 18.5,
    17.5, 16.5, 15.5, 14.3, 13.0,  8.5,  8.2,  9.5, 10.5, 12.0,
    13.5, 15.0, 16.0, 17.5, 18.5, 20.0, 21.0, 22.5, 24.0, 24.5,
    24.0, 25.5, 26.5, 26.5, 26.8, 27.3, 28.0, 28.5, 29.5, 30.0,
    30.5, 29.5, 28.5, 27.5, 27.0, 26.5, 26.5, 26.5, 26.5, 27.5,
    28.5, 29.0, 30.0, 30.5, 31.0, 30.5, 30.0, 30.0, 30.5, 31.0,
    31.0, 30.5, 30.5, 32.0, 33.0, 33.5, 32.5, 33.0, 33.5, 33.5,
    34.5, 34.8, 34.0, 33.5, 32.5, 31.5, 30.5, 29.5, 28.5, 22.5,
]

# ── WMO weather code → human-readable condition ───────────────────────────────
#   Open-Meteo uses WMO standard numeric codes. We convert them to English.
def _wmo_condition(code: int) -> str:
    if code == 0:              return "Clear"
    if code in (1, 2):         return "Clouds"
    if code == 3:              return "Overcast"
    if code in (45, 48):       return "Fog"
    if code in (51, 53, 55):   return "Drizzle"
    if code in (61, 63, 65):   return "Rain"
    if code in (71, 73, 75):   return "Snow"
    if code in (80, 81, 82):   return "Rain"
    if code in (95, 96, 99):   return "Thunderstorm"
    return "Clouds"

# ── Condition → emoji (used in the left panel icon) ───────────────────────────
ICONS = {
    "Clear": "☀️",  "Clouds": "☁️",   "Overcast": "☁️",
    "Rain":  "🌧️",  "Drizzle": "🌦️", "Thunderstorm": "⛈️",
    "Snow":  "❄️",  "Fog": "🌫️",     "default": "🌡️",
}


# ══════════════════════════════════════════════════════════════════════════════
#  AQI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def pm25_to_aqi(pm25: float) -> tuple[int, str, str]:
    """
    Convert PM2.5 concentration (µg/m³) to AQI number + category.

    This is the standard EPA formula:
        AQI = ((AQI_high - AQI_low) / (PM_high - PM_low)) * (PM - PM_low) + AQI_low

    Think of it like converting a test score to a letter grade —
    raw PM2.5 is the score, AQI is the grade on a 0-500 scale.

    Returns: (aqi_int, category_label, color_key)
    """
    pm25 = max(0.0, float(pm25))
    for pm_low, pm_high, aqi_low, aqi_high, label, color_key in AQI_BREAKPOINTS:
        if pm_low <= pm25 <= pm_high:
            # Linear interpolation within the breakpoint range
            aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
            return int(round(aqi)), label, color_key
    # If PM2.5 is extremely high (> 500 µg/m³), clamp to hazardous
    return 500, "Hazardous", "aqi_hazardous"


def aqi_color(aqi: int) -> str:
    """Return a hex color string for an AQI value (used in charts)."""
    if aqi <= 50:   return C["aqi_good"]
    if aqi <= 100:  return C["aqi_moderate"]
    if aqi <= 150:  return C["aqi_sensitive"]
    if aqi <= 200:  return C["aqi_unhealthy"]
    if aqi <= 300:  return C["aqi_very"]
    return C["aqi_hazardous"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — WeatherService
#  Fetches data from Open-Meteo (weather + air quality)
# ══════════════════════════════════════════════════════════════════════════════

class WeatherService:
    """
    This class is our "data fetcher". It knows how to talk to the
    Open-Meteo API and translate the response into Python dictionaries
    that the rest of the app can use.

    Main methods:
      get_current(city)      → current weather snapshot
      get_historical(city)   → 7-day hourly observations (for ML training)
      get_aqi(city)          → 7-day hourly air quality observations
      get_all_cities()       → current weather for all 20 cities
    """

    # ── Fetch current weather for one city ───────────────────────────────────
    def get_current(self, city: str) -> dict:
        """
        Ask Open-Meteo: 'what's the weather like RIGHT NOW in [city]?'
        Returns a dict with temperature, humidity, wind, etc.
        """
        lat, lon = CITIES[city]
        params = {
            "latitude":  lat,
            "longitude": lon,
            # 'current' means: give me values for this exact moment
            "current":   ("temperature_2m,relative_humidity_2m,"
                          "apparent_temperature,surface_pressure,"
                          "wind_speed_10m,weather_code,visibility"),
            "wind_speed_unit": "kmh",
            "timezone":  "Asia/Kolkata",
        }
        try:
            r = requests.get(OPENMETEO_URL, params=params, timeout=10)
            if r.status_code == 200:
                return self._parse_current(city, r.json())
        except Exception:
            pass
        # If the internet is down or the API fails, use a physics-based estimate
        return self._fallback_current(city)

    # ── Fetch 7 days of real hourly weather observations ─────────────────────
    def get_historical(self, city: str) -> list[dict]:
        """
        Ask Open-Meteo: 'what were the ACTUAL temperatures every hour
        for the past 7 days in [city]?'

        WHY DO WE NEED 7 DAYS?
        ───────────────────────
        Our ML model needs to see multiple full cycles (multiple days)
        to reliably estimate the wave pattern. With only 1 day, the
        sine fit could go wrong. With 7 days (168 data points),
        the model locks onto the true daily rhythm very confidently.
        """
        lat, lon = CITIES[city]
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "wind_speed_unit": "kmh",
            "past_days":  7,        # include 7 days of historical data
            "forecast_days": 1,     # also include today up to the current hour
            "timezone":   "Asia/Kolkata",
        }
        try:
            r = requests.get(OPENMETEO_URL, params=params, timeout=10)
            if r.status_code == 200:
                return self._parse_historical(r.json())
        except Exception:
            pass
        return self._fallback_historical(city)

    # ── Fetch 7 days of real hourly air quality data ──────────────────────────
    def get_aqi(self, city: str) -> list[dict]:
        """
        Ask Open-Meteo's air quality endpoint: 'what were the actual
        pollution levels every hour for the past 7 days in [city]?'

        We request:
          pm2_5   → Fine particles (most harmful, from cars/factories/fires)
          pm10    → Coarser particles (dust, pollen)
          ozone   → Ground-level ozone (from sunlight + car exhaust)
          nitrogen_dioxide → NO2 (from combustion engines)

        All of these are used to compute the AQI.
        """
        lat, lon = CITIES[city]
        params = {
            "latitude":     lat,
            "longitude":    lon,
            "hourly":       "pm2_5,pm10,ozone,nitrogen_dioxide",
            "past_days":    7,
            "forecast_days": 1,
            "timezone":     "Asia/Kolkata",
        }
        try:
            r = requests.get(OPENMETEO_AQI_URL, params=params, timeout=10)
            if r.status_code == 200:
                return self._parse_aqi(r.json())
        except Exception:
            pass
        return self._fallback_aqi(city)

    # ── Fetch current weather for all 20 Indian cities ────────────────────────
    def get_all_cities(self) -> dict[str, dict]:
        """Loop through every city and get current conditions. Used for the heatmap."""
        return {city: self.get_current(city) for city in CITIES}

    # ── Parse the JSON response from Open-Meteo (current weather) ─────────────
    def _parse_current(self, city: str, d: dict) -> dict:
        cur  = d["current"]
        code = int(cur.get("weather_code", 0))
        cond = _wmo_condition(code)
        vis_raw = cur.get("visibility", 10000)
        # Visibility sometimes comes in metres, sometimes in km — handle both
        vis_km = round(float(vis_raw) / 1000, 1) if float(vis_raw) > 100 else round(float(vis_raw), 1)
        return {
            "city":       city,
            "temp":       round(float(cur["temperature_2m"]), 1),
            "feels_like": round(float(cur["apparent_temperature"]), 1),
            "humidity":   int(cur["relative_humidity_2m"]),
            "wind_speed": round(float(cur["wind_speed_10m"]), 1),
            "condition":  cond,
            "desc":       cond,
            "pressure":   int(cur.get("surface_pressure", 1013)),
            "visibility": vis_km,
            "lat":        CITIES[city][0],
            "lon":        CITIES[city][1],
        }

    # ── Parse the JSON response from Open-Meteo (hourly historical) ───────────
    def _parse_historical(self, d: dict) -> list[dict]:
        """
        Convert the raw hourly JSON into a clean list of dicts.
        We throw away any future hours — we only want past observations.
        Future hours haven't happened yet, so they're model forecasts
        (not real measurements). We don't want to train on those.
        """
        hourly = d["hourly"]
        times  = hourly["time"]
        temps  = hourly["temperature_2m"]
        humids = hourly["relative_humidity_2m"]
        winds  = hourly["wind_speed_10m"]

        now = datetime.now()
        out = []
        for t_str, temp, hum, wind in zip(times, temps, humids, winds):
            dt = datetime.fromisoformat(t_str)
            if dt > now:       # skip future hours
                continue
            if temp is None:   # skip missing readings
                continue
            out.append({
                "datetime":   dt,
                "temp":       round(float(temp), 1),
                "humidity":   int(hum)   if hum  is not None else 60,
                "wind_speed": round(float(wind), 1) if wind is not None else 10.0,
                "condition":  "Clear",
            })
        return out

    # ── Parse the JSON response from Open-Meteo (AQI hourly) ─────────────────
    def _parse_aqi(self, d: dict) -> list[dict]:
        """
        Convert raw pollution data into AQI values.

        For each hour, we:
          1. Get the PM2.5 reading
          2. Convert it to AQI using the EPA breakpoint formula
          3. Store everything in a dict
        """
        hourly = d["hourly"]
        times  = hourly.get("time", [])
        pm25s  = hourly.get("pm2_5", [])
        pm10s  = hourly.get("pm10", [])
        ozones = hourly.get("ozone", [])
        no2s   = hourly.get("nitrogen_dioxide", [])

        now = datetime.now()
        out = []
        for t_str, pm25, pm10, oz, no2 in zip(times, pm25s, pm10s, ozones, no2s):
            dt = datetime.fromisoformat(t_str)
            if dt > now:    # only real observations
                continue
            if pm25 is None:
                continue

            aqi_val, aqi_label, aqi_clr = pm25_to_aqi(pm25 or 0)
            out.append({
                "datetime":  dt,
                "pm25":      round(float(pm25),  1) if pm25  is not None else 0.0,
                "pm10":      round(float(pm10),  1) if pm10  is not None else 0.0,
                "ozone":     round(float(oz),    1) if oz    is not None else 0.0,
                "no2":       round(float(no2),   1) if no2   is not None else 0.0,
                "aqi":       aqi_val,
                "label":     aqi_label,
                "color_key": aqi_clr,
            })
        return out

    # ── Fallback: physics-based estimates when internet is unavailable ─────────
    def _fallback_current(self, city: str) -> dict:
        """
        If the API call fails, we still want to show something reasonable.
        This generates synthetic weather based on:
          - The city's latitude (northern cities are cooler)
          - The current month (summer vs winter)
          - A random seed based on the city name (so it's consistent)
        """
        lat, lon = CITIES[city]
        month    = datetime.now().month
        rng      = np.random.default_rng(seed=hash(city) % 9999 + month)

        # Temperature decreases as you go north (~0.5°C per degree of latitude)
        base = 30 - (lat - 10) * 0.5
        seasonal = {1:-5,2:-3,3:2,4:8,5:12,6:10,7:5,8:4,9:3,10:0,11:-4,12:-6}
        temp = base + seasonal.get(month, 0) + rng.normal(0, 2)

        # Hill stations are always cooler
        if city in {"Shimla","Manali","Nainital","Dehradun","Rishikesh"}:
            temp -= 8

        cond = rng.choice(["Clear","Clouds","Haze","Rain"],
                           p=[0.1,0.2,0.1,0.6] if month in {6,7,8}
                             else [0.5,0.25,0.15,0.1])
        return {
            "city": city, "temp": round(float(temp), 1),
            "feels_like": round(float(temp) - 2 + rng.normal(0, 1), 1),
            "humidity":  int(rng.integers(40, 86)),
            "wind_speed": round(float(rng.uniform(5, 30)), 1),
            "condition": cond, "desc": cond,
            "pressure":  int(rng.integers(1005, 1021)),
            "visibility": round(float(rng.uniform(5, 15)), 1),
            "lat": lat, "lon": lon,
        }

    def _fallback_historical(self, city: str) -> list[dict]:
        """
        Generate 7 days of synthetic hourly data when the API is unavailable.
        Uses a sine wave so the ML model still has a sensible pattern to fit.
        """
        cur  = self._fallback_current(city)
        base = cur["temp"]
        rng  = np.random.default_rng(seed=hash(city) % 7777)
        now  = datetime.now()
        out  = []
        for i in range(7 * 24):
            dt   = now - timedelta(hours=167 - i)
            hour = dt.hour
            # Daily cycle: coldest around 5am, warmest around 2pm
            daily = 6.5 * math.sin(math.pi * (hour - 5) / 12)
            temp  = base + daily + rng.normal(0, 0.6)
            out.append({
                "datetime":   dt,
                "temp":       round(float(temp), 1),
                "humidity":   int(np.clip(cur["humidity"] + rng.normal(0, 8), 20, 100)),
                "wind_speed": round(max(0.0, cur["wind_speed"] + float(rng.normal(0, 3))), 1),
                "condition":  cur["condition"],
            })
        return out

    def _fallback_aqi(self, city: str) -> list[dict]:
        """
        Generate synthetic AQI data when the API is unavailable.
        Delhi-class cities get higher pollution levels than coastal/hill cities.
        AQI follows a daily pattern too: higher in morning/evening rush hours.
        """
        # Rough base PM2.5 by city type
        high_pollution = {"Delhi","Lucknow","Patna","Varanasi","Jaipur","Agra","Kanpur"}
        moderate_pollution = {"Mumbai","Kolkata","Hyderabad","Ahmedabad","Chandigarh","Guwahati","Bhubaneswar"}

        if city in high_pollution:
            base_pm25 = 80.0
        elif city in moderate_pollution:
            base_pm25 = 45.0
        else:
            base_pm25 = 20.0

        rng = np.random.default_rng(seed=hash(city) % 5555)
        now = datetime.now()
        out = []
        for i in range(7 * 24):
            dt   = now - timedelta(hours=167 - i)
            hour = dt.hour
            # Rush hour peaks: 8am and 7pm have higher pollution
            rush = 15.0 * math.exp(-((hour - 8)**2) / 8) + 12.0 * math.exp(-((hour - 19)**2) / 8)
            pm25 = max(2.0, base_pm25 + rush + rng.normal(0, 8))
            aqi_val, aqi_label, aqi_clr = pm25_to_aqi(pm25)
            out.append({
                "datetime":  dt,
                "pm25":      round(pm25, 1),
                "pm10":      round(pm25 * 1.6 + rng.normal(0, 5), 1),
                "ozone":     round(40 + rng.normal(0, 8), 1),
                "no2":       round(30 + rng.normal(0, 6), 1),
                "aqi":       aqi_val,
                "label":     aqi_label,
                "color_key": aqi_clr,
            })
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  ML LAYER — Sinusoidal Regression
#
#  THE CORE IDEA:
#  Temperature (and AQI!) follow a daily wave pattern.
#  We model this as:   Y(t) = A · sin(2π/24 · t + φ) + C
#
#  What each part means:
#    A  = Amplitude — how much does it swing up and down each day?
#         (e.g. if it goes from 20°C to 34°C, amplitude ≈ 7°C)
#    φ  = Phase — at what time of day does the peak happen?
#         (temperature peaks around 2pm → φ encodes this)
#    C  = Offset — what's the average value?
#         (e.g. if the day averages 27°C, C = 27)
#    2π/24 = How fast the wave completes one full cycle (24 hours)
#
#  scipy's curve_fit finds the A, φ, C values that make the sine wave
#  fit the 7 days of real data as closely as possible.
#  Then we just extend the wave forward 24 hours to get a prediction.
# ══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """
    A general sinusoidal regression model.

    Works for BOTH temperature and AQI — anything that has a daily cycle.

    Training takes about 1 second per city (168 data points, non-linear fitting).
    """

    def __init__(self):
        self.params:  np.ndarray | None = None  # stores the fitted (A, φ, C)
        self.sigma:   float             = 0.0   # stores the error estimate
        self.trained: bool              = False

    # ── The mathematical model: T(t) = A·sin(2π/24·t + φ) + C ────────────────
    @staticmethod
    def _sin_model(t: np.ndarray, A: float, phi: float, C: float) -> np.ndarray:
        """
        This is the actual sine wave formula.
        'static' means it doesn't need 'self' — it's a pure math function.

        t   = time in hours (0, 1, 2, 3, … 167 for 7 days)
        A   = amplitude (how tall is the wave?)
        phi = phase shift (when does the peak happen?)
        C   = vertical offset (what's the center/mean value?)
        """
        return A * np.sin((2 * np.pi / 24.0) * t + phi) + C

    # ── Train the model and generate 24-hour predictions ──────────────────────
    def fit_predict(self, observations: list[dict], value_key: str = "temp") -> dict:
        """
        1. Takes in a list of hourly observations (temperature OR AQI)
        2. Fits the sine wave to find the best A, φ, C
        3. Uses that wave to predict the next 24 hours

        value_key: which field to use — 'temp' for temperature, 'aqi' for AQI
        """
        # Step 1: Extract the values we want to fit
        values = np.array([o[value_key] for o in observations], dtype=float)
        hours  = np.arange(len(values), dtype=float)   # 0, 1, 2, … N-1

        # Step 2: Make smart initial guesses for A, φ, C
        #   Good initial guesses help curve_fit converge faster
        A0   = (values.max() - values.min()) / 2.0   # half the range
        phi0 = 0.0
        C0   = values.mean()

        # Set realistic bounds so the fit doesn't go crazy
        # For temperature: A in [0,30], C in [-5,55]
        # For AQI: A in [0,200], C in [0,500]
        if value_key == "temp":
            bounds = ([0.0, -np.pi, -5.0], [30.0, np.pi, 55.0])
        else:  # AQI or PM2.5
            bounds = ([0.0, -np.pi, 0.0], [300.0, np.pi, 500.0])

        try:
            self.params, _ = curve_fit(
                self._sin_model, hours, values,
                p0=[A0, phi0, C0],
                bounds=bounds,
                maxfev=20_000   # maximum iterations before giving up
            )
        except RuntimeError:
            # If fitting fails, just use our initial guesses
            self.params = np.array([A0, phi0, C0])

        self.trained = True

        # Step 3: Generate 24 future hourly predictions
        last_h     = float(len(values) - 1)
        future_h   = np.arange(last_h + 1, last_h + 25, 1.0)
        pred_vals  = self._sin_model(future_h, *self.params)

        # Ensure AQI predictions don't go negative
        if value_key in ("aqi", "pm25"):
            pred_vals = np.maximum(pred_vals, 0)

        # Step 4: Calculate confidence band
        #   We measure how much the fitted wave deviated from actual data,
        #   then use ±1.5 standard deviations as our confidence range.
        fitted_train = self._sin_model(hours, *self.params)
        residuals    = values - fitted_train
        self.sigma   = float(np.std(residuals))

        # Convert future hour indices back to real datetime objects
        t0  = observations[-1]["datetime"]
        dts = [t0 + timedelta(hours=int(i + 1)) for i in range(24)]

        return {
            "datetimes": dts,
            "values":    pred_vals,
            "lower":     pred_vals - 1.5 * self.sigma,
            "upper":     pred_vals + 1.5 * self.sigma,
            "sigma":     self.sigma,
            "params":    self.params,    # (A, φ, C)
        }

    def fitted_curve(self, observations: list[dict], value_key: str = "temp") -> tuple:
        """
        Return a smooth dense version of the fitted sine wave
        over the training window. Used for the pretty curves in charts.
        """
        if not self.trained:
            return np.array([]), np.array([])
        hours   = np.arange(len(observations), dtype=float)
        t_dense = np.linspace(hours[0], hours[-1], 600)
        return t_dense, self._sin_model(t_dense, *self.params)

    def r2(self, observations: list[dict], value_key: str = "temp") -> float:
        """
        R² (R-squared) score — how well does our model fit the data?
          R² = 1.0 → perfect fit
          R² = 0.0 → no better than just using the mean
          R² < 0.0 → the fit is worse than the mean (something went wrong)

        Anything above 0.75 is considered good for temperature prediction.
        """
        if not self.trained:
            return 0.0
        values = np.array([o[value_key] for o in observations], dtype=float)
        hours  = np.arange(len(values), dtype=float)
        fitted = self._sin_model(hours, *self.params)
        return float(round(r2_score(values, fitted), 3))


# ══════════════════════════════════════════════════════════════════════════════
#  UI LAYER — Main Application Window
# ══════════════════════════════════════════════════════════════════════════════

class WeatherApp(ctk.CTk):
    """
    The main window of the app. Built with CustomTkinter for a modern dark UI.

    Layout:
      ┌────────────────────── Header ──────────────────────────────┐
      │  Logo          City dropdown         Refresh button         │
      ├───────────────┬────────────────────────────────────────────┤
      │  Left panel   │  Tab view                                   │
      │  • City name  │  [📈 ML Forecast]  [🌫️ AQI Monitor]        │
      │  • Big temp   │  [🌡️ Heat Map]     [🏙️ City Compare]       │
      │  • Metrics    │  [📊 Analytics]                             │
      ├───────────────┴────────────────────────────────────────────┤
      │  Status bar (shows what's loading + last update time)       │
      └────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.title("🌤️  Weather + AQI Forecaster — Indian Cities (Open-Meteo)")
        self.geometry("1400x880")
        self.minsize(1100, 740)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=C["bg"])

        # Our data and ML objects
        self.svc = WeatherService()
        self.ml_temp = MLPredictor()   # one model for temperature
        self.ml_aqi  = MLPredictor()   # separate model for AQI

        # App state
        self.city          = "Dehradun"
        self.cur_weather   = None
        self.observations  = None     # temperature history (7 days)
        self.aqi_history   = None     # AQI history (7 days)
        self.all_weather   = {}       # all cities' current weather (for heatmap)
        self.loading       = False
        self._hm_cbar      = None     # colorbar reference for the heatmap

        self._build_header()
        self._build_body()
        self._build_statusbar()
        self._load_async()   # start fetching data immediately in background

    # ══════════════════════════════════════════════════════════════
    #  HEADER — top bar with title, city picker, refresh button
    # ══════════════════════════════════════════════════════════════

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C["card"], height=68, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="🌤️  Weather + AQI Forecaster — Indian Cities",
            font=ctk.CTkFont("Helvetica", 19, "bold"),
            text_color=C["accent"],
        ).pack(side="left", padx=22, pady=14)

        ctk.CTkLabel(
            hdr, text="  📡 Live: Open-Meteo API  ·  🌫️ AQI: PM2.5/PM10/O₃/NO₂",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["accent4"],
        ).pack(side="left", padx=8)

        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right", padx=20)

        ctk.CTkLabel(right, text="City:", font=ctk.CTkFont(size=13),
                     text_color=C["dim"]).pack(side="left", padx=(0, 4))

        self.city_var = ctk.StringVar(value=self.city)
        ctk.CTkComboBox(
            right, values=list(CITIES.keys()), variable=self.city_var,
            command=self._on_city_change,
            width=170, height=36,
            fg_color=C["card2"], border_color=C["accent"],
            button_color=C["accent"], button_hover_color="#0ea5e9",
            dropdown_fg_color=C["card"], dropdown_hover_color=C["card2"],
            text_color=C["text"], font=ctk.CTkFont(size=13),
        ).pack(side="left", padx=6)

        self.refresh_btn = ctk.CTkButton(
            right, text="🔄 Refresh", command=self._load_async,
            width=120, height=36,
            fg_color=C["accent2"], hover_color="#6366f1",
            text_color="white", font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=18,
        )
        self.refresh_btn.pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════════════
    #  BODY — left panel + tabbed charts
    # ══════════════════════════════════════════════════════════════

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=10)
        self._build_left_panel(body)
        self._build_tabs(body)

    # ── Left panel: current weather summary cards ─────────────────────────────

    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, fg_color="transparent", width=265)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        self.lbl_city = ctk.CTkLabel(left, text="Dehradun",
                                      font=ctk.CTkFont(size=22, weight="bold"),
                                      text_color=C["text"])
        self.lbl_city.pack(pady=(4, 0))

        self.lbl_date = ctk.CTkLabel(
            left, text=datetime.now().strftime("%A, %d %B %Y"),
            font=ctk.CTkFont(size=11), text_color=C["dim"])
        self.lbl_date.pack(pady=(0, 8))

        # Big temperature display card
        tc = ctk.CTkFrame(left, fg_color=C["card"], corner_radius=18,
                          border_width=1, border_color=C["accent"])
        tc.pack(fill="x", pady=(0, 6))

        self.lbl_icon = ctk.CTkLabel(tc, text="🌡️", font=ctk.CTkFont(size=44))
        self.lbl_icon.pack(pady=(14, 0))

        self.lbl_temp = ctk.CTkLabel(tc, text="--°C",
                                      font=ctk.CTkFont(size=40, weight="bold"),
                                      text_color=C["accent"])
        self.lbl_temp.pack()

        self.lbl_cond = ctk.CTkLabel(tc, text="Connecting to Open-Meteo…",
                                      font=ctk.CTkFont(size=12),
                                      text_color=C["dim"])
        self.lbl_cond.pack(pady=(0, 14))

        # AQI summary card in left panel
        aq = ctk.CTkFrame(left, fg_color=C["card"], corner_radius=14,
                          border_width=1, border_color=C["accent6"])
        aq.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(aq, text="🌫️  Air Quality Index",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C["accent6"]).pack(pady=(8, 0))

        self.lbl_aqi_val = ctk.CTkLabel(aq, text="AQI --",
                                         font=ctk.CTkFont(size=26, weight="bold"),
                                         text_color=C["aqi_good"])
        self.lbl_aqi_val.pack()

        self.lbl_aqi_cat = ctk.CTkLabel(aq, text="Loading…",
                                         font=ctk.CTkFont(size=11),
                                         text_color=C["dim"])
        self.lbl_aqi_cat.pack(pady=(0, 4))

        self.lbl_aqi_advice = ctk.CTkLabel(aq, text="",
                                            font=ctk.CTkFont(size=9),
                                            text_color=C["dim"],
                                            wraplength=220, justify="center")
        self.lbl_aqi_advice.pack(pady=(0, 8), padx=6)

        # Individual weather metric rows
        self._metrics = {}
        for icon, label, key, unit in [
            ("💧", "Humidity",   "humidity",   "%"),
            ("💨", "Wind Speed", "wind_speed", " km/h"),
            ("👁️", "Visibility", "visibility", " km"),
            ("🌡️", "Feels Like", "feels_like", "°C"),
            ("📊", "Pressure",   "pressure",   " hPa"),
        ]:
            row = ctk.CTkFrame(left, fg_color=C["card2"], corner_radius=12,
                               border_width=1, border_color=C["border"])
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=f"{icon}  {label}",
                         font=ctk.CTkFont(size=11), text_color=C["dim"]
                         ).pack(side="left", padx=10, pady=9)
            val = ctk.CTkLabel(row, text=f"--{unit}",
                               font=ctk.CTkFont(size=12, weight="bold"),
                               text_color=C["text"])
            val.pack(side="right", padx=10)
            self._metrics[key] = (val, unit)

        ctk.CTkLabel(
            left,
            text="🔬 ML trains on 7 days of real observed data\n"
                 "   → No forecast-on-forecast bias",
            font=ctk.CTkFont(size=9), text_color=C["dim"],
            justify="left",
        ).pack(pady=(8, 0), padx=6)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _build_tabs(self, parent):
        self.tabs = ctk.CTkTabview(
            parent, fg_color=C["card"],
            segmented_button_fg_color=C["card2"],
            segmented_button_selected_color=C["accent"],
            segmented_button_unselected_color=C["card2"],
            segmented_button_selected_hover_color="#0ea5e9",
            text_color=C["text"], corner_radius=16,
        )
        self.tabs.pack(side="right", fill="both", expand=True)
        tab_names = [
            "📈 ML Forecast",
            "🌫️ AQI Monitor",
            "🌡️ Heat Map",
            "🏙️ City Compare",
            "📊 Analytics",
        ]
        for name in tab_names:
            self.tabs.add(name)

        self._build_forecast_tab  (self.tabs.tab("📈 ML Forecast"))
        self._build_aqi_tab       (self.tabs.tab("🌫️ AQI Monitor"))
        self._build_heatmap_tab   (self.tabs.tab("🌡️ Heat Map"))
        self._build_compare_tab   (self.tabs.tab("🏙️ City Compare"))
        self._build_analytics_tab (self.tabs.tab("📊 Analytics"))

    # ── TAB 1: ML Temperature Forecast ────────────────────────────────────────

    def _build_forecast_tab(self, parent):
        info = ctk.CTkFrame(parent, fg_color="transparent")
        info.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(
            info,
            text="〰️  Model: T(t) = A·sin(2π/24·t + φ) + C   "
                 "Trained on 7 days of REAL hourly temperature observations",
            font=ctk.CTkFont(size=10), text_color=C["dim"],
        ).pack(side="left")
        self.lbl_score = ctk.CTkLabel(info, text="R²=--",
                                       font=ctk.CTkFont(size=10, weight="bold"),
                                       text_color=C["accent4"])
        self.lbl_score.pack(side="right")

        self.fig_fc  = Figure(figsize=(8, 5), facecolor=C["card"])
        self.ax_fc   = self.fig_fc.add_subplot(111)
        self._style_ax(self.ax_fc)
        self.canvas_fc = FigureCanvasTkAgg(self.fig_fc, parent)
        self.canvas_fc.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ── TAB 2: AQI Monitor ────────────────────────────────────────────────────

    def _build_aqi_tab(self, parent):
        # Top info bar
        info = ctk.CTkFrame(parent, fg_color="transparent")
        info.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(
            info,
            text="🌫️  AQI Model: AQI(t) = A·sin(2π/24·t + φ) + C   "
                 "Trained on 7 days of real PM2.5 observations (Open-Meteo AQ API)",
            font=ctk.CTkFont(size=10), text_color=C["dim"],
        ).pack(side="left")
        self.lbl_aqi_score = ctk.CTkLabel(info, text="R²=--",
                                           font=ctk.CTkFont(size=10, weight="bold"),
                                           text_color=C["accent6"])
        self.lbl_aqi_score.pack(side="right")

        # The AQI chart: 2 rows × 1 column
        # Top: AQI time series + 24h prediction
        # Bottom: PM2.5 / PM10 / Ozone / NO2 breakdown
        self.fig_aqi = Figure(figsize=(8, 5.5), facecolor=C["card"])
        self.ax_aqi1 = self.fig_aqi.add_subplot(211)   # main AQI line
        self.ax_aqi2 = self.fig_aqi.add_subplot(212)   # pollutant breakdown
        self._style_ax(self.ax_aqi1)
        self._style_ax(self.ax_aqi2)
        self.fig_aqi.tight_layout(pad=2.5)
        self.canvas_aqi = FigureCanvasTkAgg(self.fig_aqi, parent)
        self.canvas_aqi.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ── TAB 3: Heat Map ───────────────────────────────────────────────────────

    def _build_heatmap_tab(self, parent):
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(fill="x", padx=10, pady=(4, 0))

        ctk.CTkLabel(ctrl, text="Variable:", text_color=C["dim"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))

        self.heat_var = ctk.StringVar(value="Temperature")
        for opt in ["Temperature", "Humidity", "Wind Speed"]:
            ctk.CTkRadioButton(
                ctrl, text=opt, variable=self.heat_var, value=opt,
                command=self._draw_heatmap,
                text_color=C["text"], fg_color=C["accent"], hover_color="#0ea5e9",
            ).pack(side="left", padx=8)

        ctk.CTkLabel(ctrl, text="  Interpolation:", text_color=C["dim"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(14, 4))
        self.interp_var = ctk.StringVar(value="cubic")
        for m in ["linear", "cubic", "nearest"]:
            ctk.CTkRadioButton(
                ctrl, text=m, variable=self.interp_var, value=m,
                command=self._draw_heatmap,
                text_color=C["text"], fg_color=C["accent2"], hover_color="#6366f1",
            ).pack(side="left", padx=6)

        self.fig_hm  = Figure(figsize=(8, 5.2), facecolor=C["card"])
        self.ax_hm   = self.fig_hm.add_subplot(111)
        self._style_ax(self.ax_hm)
        self.canvas_hm = FigureCanvasTkAgg(self.fig_hm, parent)
        self.canvas_hm.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ── TAB 4: City Compare ───────────────────────────────────────────────────

    def _build_compare_tab(self, parent):
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(fill="x", padx=10, pady=(4, 0))
        ctk.CTkLabel(ctrl, text="Cities (comma-separated):",
                     text_color=C["dim"], font=ctk.CTkFont(size=12)
                     ).pack(side="left", padx=(0, 6))
        self.compare_var = ctk.StringVar(
            value="Dehradun, Delhi, Mumbai, Bangalore, Shimla, Chennai")
        ctk.CTkEntry(ctrl, textvariable=self.compare_var,
                     width=370, height=32,
                     fg_color=C["card2"], border_color=C["border"],
                     text_color=C["text"], font=ctk.CTkFont(size=12)
                     ).pack(side="left", padx=6)
        ctk.CTkButton(ctrl, text="Update", command=self._draw_compare,
                      width=80, height=32,
                      fg_color=C["accent4"], hover_color="#059669",
                      text_color=C["bg"], font=ctk.CTkFont(size=12, weight="bold"),
                      corner_radius=8).pack(side="left", padx=4)

        self.fig_cmp  = Figure(figsize=(8, 5), facecolor=C["card"])
        self.ax_cmp   = self.fig_cmp.add_subplot(111)
        self._style_ax(self.ax_cmp)
        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, parent)
        self.canvas_cmp.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ── TAB 5: Analytics Dashboard ────────────────────────────────────────────

    def _build_analytics_tab(self, parent):
        # 2×2 grid of small charts
        self.fig_an = Figure(figsize=(8, 5), facecolor=C["card"])
        self.ax_an1 = self.fig_an.add_subplot(221)  # humidity over time
        self.ax_an2 = self.fig_an.add_subplot(222)  # wind speed over time
        self.ax_an3 = self.fig_an.add_subplot(223)  # condition pie chart
        self.ax_an4 = self.fig_an.add_subplot(224)  # temperature ranking
        for ax in (self.ax_an1, self.ax_an2, self.ax_an3, self.ax_an4):
            self._style_ax(ax)
        self.fig_an.tight_layout(pad=2.2)
        self.canvas_an = FigureCanvasTkAgg(self.fig_an, parent)
        self.canvas_an.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color=C["card"], height=26, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.lbl_status = ctk.CTkLabel(bar, text="Ready",
                                        font=ctk.CTkFont(size=10),
                                        text_color=C["dim"])
        self.lbl_status.pack(side="left", padx=14)
        ctk.CTkLabel(
            bar,
            text="Weather+AQI: Open-Meteo (real observed)  |  ML: scipy sinusoidal  |  Heatmap: scipy griddata",
            font=ctk.CTkFont(size=10), text_color=C["dim"],
        ).pack(side="right", padx=14)

    # ══════════════════════════════════════════════════════════════
    #  STYLING HELPER — applies consistent dark theme to all charts
    # ══════════════════════════════════════════════════════════════

    def _style_ax(self, ax, title=""):
        ax.set_facecolor(C["card2"])
        ax.tick_params(colors=C["dim"], labelsize=8)
        ax.xaxis.label.set_color(C["dim"])
        ax.yaxis.label.set_color(C["dim"])
        if title:
            ax.set_title(title, color=C["text"], fontsize=11, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.grid(True, color=C["border"], alpha=0.45, linewidth=0.5)

    # ══════════════════════════════════════════════════════════════
    #  DATA LOADING — runs in a background thread so the UI stays responsive
    # ══════════════════════════════════════════════════════════════

    def _load_async(self):
        """
        Called when user hits Refresh or changes city.
        We spawn a background thread so the window doesn't freeze
        while we wait for the API response.
        """
        if self.loading:
            return
        self.loading = True
        self.refresh_btn.configure(state="disabled", text="⏳ Loading…")
        self._status("Connecting to Open-Meteo…")
        threading.Thread(target=self._load_thread, daemon=True).start()

    def _load_thread(self):
        """
        This runs in a background thread.
        We fetch data in order: current → historical → AQI → all cities.
        Each step updates the status bar so the user knows what's happening.
        """
        try:
            self._status(f"Fetching current weather for {self.city}…")
            self.cur_weather = self.svc.get_current(self.city)

            self._status(f"Fetching 7-day temperature observations for {self.city}…")
            self.observations = self.svc.get_historical(self.city)

            self._status(f"Fetching 7-day AQI observations for {self.city}…")
            self.aqi_history = self.svc.get_aqi(self.city)

            self._status("Fetching current conditions for all 20 Indian cities…")
            self.all_weather = self.svc.get_all_cities()

            # Schedule UI update back on the main thread
            # (Tkinter requires all UI updates to happen on the main thread)
            self.after(0, self._refresh_all)
            self._status(
                f"✓  Updated at {datetime.now().strftime('%H:%M:%S')}  "
                f"({len(self.observations)} temp obs, {len(self.aqi_history)} AQI obs)"
            )
        except Exception as ex:
            self._status(f"⚠  Error: {ex}")
        finally:
            self.loading = False
            self.after(0, lambda: self.refresh_btn.configure(
                state="normal", text="🔄 Refresh"))

    def _refresh_all(self):
        """Called on the main thread once all data is loaded. Redraws everything."""
        self._update_left_panel()
        self._draw_forecast()
        self._draw_aqi()
        self._draw_heatmap()
        self._draw_compare()
        self._draw_analytics()

    # ══════════════════════════════════════════════════════════════
    #  UI UPDATE METHODS
    # ══════════════════════════════════════════════════════════════

    def _update_left_panel(self):
        """Update the left panel cards with current weather + AQI."""
        w = self.cur_weather
        if not w:
            return

        # Update weather section
        self.lbl_city.configure(text=w["city"])
        self.lbl_cond.configure(text=w["desc"])
        self.lbl_icon.configure(text=ICONS.get(w["condition"], ICONS["default"]))

        # Color the temperature based on how hot it is
        t = w["temp"]
        color = (C["accent4"] if t < 15 else
                 C["accent"]  if t < 25 else
                 C["warning"] if t < 35 else C["accent3"])
        self.lbl_temp.configure(text=f"{t}°C", text_color=color)

        for key, (lbl, unit) in self._metrics.items():
            lbl.configure(text=f"{w.get(key, '--')}{unit}")

        # Update AQI section using the most recent AQI observation
        if self.aqi_history:
            latest = self.aqi_history[-1]
            aqi_val = latest["aqi"]
            aqi_lbl = latest["label"]
            aqi_col = C[latest["color_key"]]
            advice  = AQI_ADVICE.get(aqi_lbl, "")
            self.lbl_aqi_val.configure(text=f"AQI {aqi_val}", text_color=aqi_col)
            self.lbl_aqi_cat.configure(text=aqi_lbl, text_color=aqi_col)
            self.lbl_aqi_advice.configure(text=advice)

    # ── CHART 1: Temperature Forecast ─────────────────────────────────────────

    def _draw_forecast(self):
        """
        Draws 3 layers on one chart:
          Layer 1 (grey dots)  → 7 days of real hourly temperature measurements
          Layer 2 (pink line)  → the fitted sine wave drawn over the training data
          Layer 3 (orange dash)→ the predicted next 24 hours (same wave, extended)
                                 + shaded ±1.5σ confidence band

        A text box in the corner shows the fitted equation with actual numbers.
        """
        if not self.observations:
            return

        pred  = self.ml_temp.fit_predict(self.observations, "temp")
        score = self.ml_temp.r2(self.observations, "temp")
        A, phi, Cval = pred["params"]

        ax = self.ax_fc
        ax.clear()
        self._style_ax(ax, f"📈 Sinusoidal ML Temperature Forecast — {self.city}  "
                           f"({len(self.observations)} real hourly observations)")

        # Layer 1: scatter plot of observed data points
        obs_times = [o["datetime"] for o in self.observations]
        obs_temps = [o["temp"]     for o in self.observations]
        ax.scatter(obs_times, obs_temps,
                   color=C["dim"], s=6, alpha=0.50, zorder=3,
                   label=f"Observed (real 7-day data, {len(self.observations)} pts)")

        # Layer 2: smooth fitted sine wave over training window
        t_dense, y_dense = self.ml_temp.fitted_curve(self.observations, "temp")
        if len(t_dense) > 0:
            origin    = obs_times[0]
            dense_dts = [origin + timedelta(hours=float(h)) for h in t_dense]
            ax.plot(dense_dts, y_dense,
                    color=C["accent5"], linewidth=2.0, alpha=0.90, zorder=4,
                    label="Fitted sine  T(t) = A·sin(2π/24·t + φ) + C")

        # Layer 3: dashed prediction line + confidence shading
        ml_times = pred["datetimes"]
        ml_vals  = pred["values"]
        ax.plot(ml_times, ml_vals,
                color=C["accent3"], linewidth=2.4,
                linestyle="--", marker="s", markersize=4, zorder=5,
                label="Prediction (next 24 h)")
        ax.fill_between(ml_times, pred["lower"], pred["upper"],
                        alpha=0.22, color=C["accent3"],
                        label=f"±1.5σ confidence  (σ = {pred['sigma']:.2f}°C)")

        # Vertical "now" line
        ax.axvline(datetime.now(), color=C["accent4"],
                   linestyle=":", linewidth=1.8, label="Now")

        # Equation box showing fitted parameters
        eq = (f"Fitted equation:\n"
              f"T(t) = {A:.2f} · sin(2π/24 · t + {phi:.2f}) + {Cval:.1f}\n"
              f"Amplitude = {A:.2f}°C   Period = 24h   Mean = {Cval:.1f}°C")
        ax.text(0.01, 0.97, eq, transform=ax.transAxes,
                fontsize=8.5, verticalalignment="top", color=C["text"],
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=C["card2"], alpha=0.88,
                          edgecolor=C["accent2"]))

        ax.set_xlabel("Date / Time")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(facecolor=C["card2"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=8, loc="upper right")
        self.fig_fc.autofmt_xdate()
        self.fig_fc.tight_layout()
        self.canvas_fc.draw()

        self.lbl_score.configure(
            text=f"R² = {score:.3f}   A = {A:.1f}°C   φ = {phi:.2f} rad   C = {Cval:.1f}°C")

    # ── CHART 2: AQI Monitor ──────────────────────────────────────────────────

    def _draw_aqi(self):
        """
        Two-panel AQI chart:

        TOP PANEL:
          • Grey dots    → 7 days of real hourly AQI observations
          • Violet line  → fitted sine wave over training data
          • Dashed line  → 24-hour AQI prediction
          • AQI zone bands → color-coded horizontal bands (Good/Moderate/etc.)

        BOTTOM PANEL:
          • 4 area charts showing PM2.5, PM10, Ozone, NO2 over the 7 days
          These let you see which pollutant is driving the AQI up.
        """
        if not self.aqi_history:
            return

        # ── Fit the sinusoidal model to AQI values ──────────────────────────
        pred  = self.ml_aqi.fit_predict(self.aqi_history, "aqi")
        score = self.ml_aqi.r2(self.aqi_history, "aqi")
        A, phi, Cval = pred["params"]

        # ── TOP PANEL: AQI time series + prediction ─────────────────────────
        ax1 = self.ax_aqi1
        ax1.clear()
        self._style_ax(ax1, f"🌫️  AQI Forecast — {self.city}  "
                           f"(PM2.5-based, {len(self.aqi_history)} observations)")

        # Draw AQI health zone bands (background shading)
        zone_ranges = [
            (0,   50,  C["aqi_good"],      "Good (0-50)"),
            (50,  100, C["aqi_moderate"],  "Moderate (51-100)"),
            (100, 150, C["aqi_sensitive"], "Sensitive (101-150)"),
            (150, 200, C["aqi_unhealthy"], "Unhealthy (151-200)"),
            (200, 300, C["aqi_very"],      "Very Unhealthy (201-300)"),
            (300, 500, C["aqi_hazardous"], "Hazardous (301+)"),
        ]
        for lo, hi, clr, _ in zone_ranges:
            ax1.axhspan(lo, hi, alpha=0.07, color=clr, zorder=1)

        # Observed AQI dots
        obs_times = [o["datetime"] for o in self.aqi_history]
        obs_aqis  = [o["aqi"]      for o in self.aqi_history]

        # Color each dot by its AQI category
        dot_colors = [aqi_color(v) for v in obs_aqis]
        ax1.scatter(obs_times, obs_aqis,
                    c=dot_colors, s=7, alpha=0.60, zorder=3,
                    label=f"Observed AQI ({len(self.aqi_history)} pts)")

        # Smooth fitted AQI sine wave
        t_dense, y_dense = self.ml_aqi.fitted_curve(self.aqi_history, "aqi")
        if len(t_dense) > 0:
            origin    = obs_times[0]
            dense_dts = [origin + timedelta(hours=float(h)) for h in t_dense]
            y_dense_clipped = np.maximum(y_dense, 0)
            ax1.plot(dense_dts, y_dense_clipped,
                     color=C["accent6"], linewidth=2.0, alpha=0.90, zorder=4,
                     label="Fitted sine  AQI(t) = A·sin(2π/24·t + φ) + C")

        # Dashed prediction + confidence band
        ml_times = pred["datetimes"]
        ml_aqi_v = np.maximum(pred["values"], 0)
        ax1.plot(ml_times, ml_aqi_v,
                 color=C["warning"], linewidth=2.4,
                 linestyle="--", marker="D", markersize=3.5, zorder=5,
                 label="AQI Prediction (next 24 h)")
        ax1.fill_between(ml_times,
                         np.maximum(pred["lower"], 0),
                         np.maximum(pred["upper"], 0),
                         alpha=0.20, color=C["warning"],
                         label=f"±1.5σ confidence (σ={pred['sigma']:.1f})")

        # "Now" line
        ax1.axvline(datetime.now(), color=C["accent4"],
                    linestyle=":", linewidth=1.8, label="Now")

        # AQI zone legend patches
        legend_patches = [
            mpatches.Patch(color=C["aqi_good"],      label="Good"),
            mpatches.Patch(color=C["aqi_moderate"],  label="Moderate"),
            mpatches.Patch(color=C["aqi_sensitive"], label="Sensitive"),
            mpatches.Patch(color=C["aqi_unhealthy"], label="Unhealthy"),
            mpatches.Patch(color=C["aqi_very"],      label="Very Unhealthy"),
            mpatches.Patch(color=C["aqi_hazardous"], label="Hazardous"),
        ]
        ax1.legend(handles=legend_patches,
                   facecolor=C["card2"], edgecolor=C["border"],
                   labelcolor=C["text"], fontsize=7, loc="upper right",
                   ncol=3)

        # Equation annotation
        eq = (f"AQI(t) = {A:.1f}·sin(2π/24·t + {phi:.2f}) + {Cval:.0f}\n"
              f"Amplitude = {A:.1f}   Mean AQI = {Cval:.0f}")
        ax1.text(0.01, 0.97, eq, transform=ax1.transAxes,
                 fontsize=8.5, verticalalignment="top", color=C["text"],
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor=C["card2"], alpha=0.88,
                           edgecolor=C["accent6"]))

        ax1.set_ylabel("AQI (0–500)")
        ax1.set_ylim(bottom=0)
        self.fig_aqi.autofmt_xdate()

        # ── BOTTOM PANEL: individual pollutant breakdown ─────────────────────
        ax2 = self.ax_aqi2
        ax2.clear()
        self._style_ax(ax2, "🔬 Pollutant Breakdown — PM2.5  PM10  O₃  NO₂")

        # Extract each pollutant as a separate time series
        pm25_vals = [o["pm25"]  for o in self.aqi_history]
        pm10_vals = [o["pm10"]  for o in self.aqi_history]
        oz_vals   = [o["ozone"] for o in self.aqi_history]
        no2_vals  = [o["no2"]   for o in self.aqi_history]

        # Stacked area chart — lets you see the contribution of each pollutant
        ax2.fill_between(obs_times, pm25_vals,
                         alpha=0.45, color=C["error"],    label="PM2.5 (µg/m³)")
        ax2.fill_between(obs_times, pm10_vals,
                         alpha=0.30, color=C["warning"],  label="PM10  (µg/m³)")
        ax2.fill_between(obs_times, oz_vals,
                         alpha=0.30, color=C["accent2"],  label="O₃    (µg/m³)")
        ax2.fill_between(obs_times, no2_vals,
                         alpha=0.30, color=C["accent4"],  label="NO₂   (µg/m³)")

        ax2.plot(obs_times, pm25_vals, color=C["error"],   linewidth=1.3, alpha=0.9)
        ax2.plot(obs_times, pm10_vals, color=C["warning"], linewidth=1.0, alpha=0.7)

        ax2.set_ylabel("Concentration (µg/m³)")
        ax2.set_xlabel("Date / Time")
        ax2.legend(facecolor=C["card2"], edgecolor=C["border"],
                   labelcolor=C["text"], fontsize=8, loc="upper right", ncol=2)

        self.fig_aqi.tight_layout(pad=2.0)
        self.canvas_aqi.draw()

        # Update the score label in the tab header
        self.lbl_aqi_score.configure(
            text=f"R² = {score:.3f}   A = {A:.1f}   φ = {phi:.2f}   C = {Cval:.0f}")

    # ── CHART 3: Geographic Heatmap ───────────────────────────────────────────

    def _draw_heatmap(self):
        """
        Creates a smooth color-gradient map of India.

        The tricky part: we only have 20 city data points (dots on a map),
        but we want a continuous smooth color field. We solve this with
        spatial interpolation using scipy.griddata:

          1. Place our 20 city values on their lat/lon positions
          2. Create a 300×300 grid covering India
          3. Use griddata to estimate values at every grid point
          4. Draw filled contours (contourf) using those estimated values

        This transforms 20 dots into a full-coverage gradient map.
        """
        if not self.all_weather:
            return

        metric = self.heat_var.get()
        method = self.interp_var.get()
        ax     = self.ax_hm
        ax.clear()

        # Remove old colorbar before drawing a new one
        if self._hm_cbar is not None:
            try:
                self._hm_cbar.remove()
            except Exception:
                pass
            self._hm_cbar = None

        # Collect city positions and the selected metric value
        lats, lons, vals, names = [], [], [], []
        for city, w in self.all_weather.items():
            lat, lon = CITIES[city]
            lats.append(lat); lons.append(lon); names.append(city)
            if metric == "Temperature":
                vals.append(w["temp"])
            elif metric == "Humidity":
                vals.append(w["humidity"])
            else:
                vals.append(w["wind_speed"])

        lats = np.array(lats, dtype=float)
        lons = np.array(lons, dtype=float)
        vals = np.array(vals, dtype=float)

        # Create a regular 300×300 grid covering India (lon 65-98, lat 6-38)
        GRID_N    = 300
        grid_lons = np.linspace(65.0, 98.0, GRID_N)
        grid_lats = np.linspace(6.0,  38.0, GRID_N)
        GLON, GLAT = np.meshgrid(grid_lons, grid_lats)

        # Interpolate city values onto the full grid
        # Points outside the convex hull of cities will be NaN with cubic/linear
        points   = np.column_stack((lons, lats))
        grid_val = griddata(points, vals, (GLON, GLAT), method=method)

        # Fill NaN regions (border areas) with nearest-neighbour values
        if np.any(np.isnan(grid_val)):
            grid_nn  = griddata(points, vals, (GLON, GLAT), method="nearest")
            grid_val = np.where(np.isnan(grid_val), grid_nn, grid_val)

        # Choose color map and labels for the selected metric
        cmap_name = {"Temperature": "RdYlBu_r",
                     "Humidity":    "YlGnBu",
                     "Wind Speed":  "PuRd"}[metric]
        unit      = {"Temperature": "°C",
                     "Humidity":    "%",
                     "Wind Speed":  "km/h"}[metric]
        title_str = {"Temperature": "🌡️  Temperature Heat Map — India",
                     "Humidity":    "💧  Relative Humidity — India",
                     "Wind Speed":  "💨  Wind Speed — India"}[metric]

        # Draw smooth filled contours
        levels = np.linspace(float(vals.min()), float(vals.max()), 21)
        cf = ax.contourf(GLON, GLAT, grid_val,
                         levels=levels, cmap=cmap_name, alpha=0.88, zorder=1)

        # Add subtle contour lines for definition
        ax.contour(GLON, GLAT, grid_val,
                   levels=levels[::3], colors="white",
                   linewidths=0.35, alpha=0.30, zorder=2)

        # Draw the India border outline
        ax.plot(_IND_LON + [_IND_LON[0]], _IND_LAT + [_IND_LAT[0]],
                color="#94a3b8", linewidth=1.4, linestyle="-", alpha=0.70, zorder=5)

        # Draw city markers and value labels
        sc = ax.scatter(lons, lats,
                        c=vals, cmap=cmap_name,
                        vmin=float(vals.min()), vmax=float(vals.max()),
                        s=260, zorder=6,
                        edgecolors="white", linewidths=0.9, alpha=0.95)

        for name, la, lo, v in zip(names, lats, lons, vals):
            ax.annotate(
                f"{name}\n{v:.0f}{unit}",
                xy=(lo, la), xytext=(3, 7),
                textcoords="offset points", ha="center",
                fontsize=6.5, color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          fc=C["card"], alpha=0.70, ec="none"),
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )

        # Add color bar legend
        self._hm_cbar = self.fig_hm.colorbar(cf, ax=ax, shrink=0.80, pad=0.02)
        self._hm_cbar.set_label(f"{metric}  ({unit})", color=C["dim"])
        self._hm_cbar.ax.yaxis.set_tick_params(color=C["dim"])
        plt.setp(self._hm_cbar.ax.yaxis.get_ticklabels(), color=C["dim"])

        ax.set_facecolor(C["card2"])
        ax.set_xlim(66, 98); ax.set_ylim(6, 37)
        ax.set_xlabel("Longitude", color=C["dim"])
        ax.set_ylabel("Latitude",  color=C["dim"])
        ax.tick_params(colors=C["dim"], labelsize=8)
        ax.set_title(f"{title_str}  [{method} interpolation]",
                     color=C["text"], fontsize=11, pad=6)
        ax.grid(True, color=C["border"], alpha=0.25, linewidth=0.4, zorder=0)
        ax.text(0.01, 0.01,
                f"Data: Open-Meteo  ·  Interpolation: scipy.griddata ({method})",
                transform=ax.transAxes, fontsize=7, color=C["dim"], va="bottom")

        self.fig_hm.tight_layout()
        self.canvas_hm.draw()

    # ── CHART 4: City Compare ─────────────────────────────────────────────────

    def _draw_compare(self):
        """
        Side-by-side bar chart comparing multiple cities.
        Shows temperature, humidity, and wind speed for each city.
        """
        if not self.all_weather:
            return

        raw    = self.compare_var.get()
        cities = [c.strip() for c in raw.split(",") if c.strip() in CITIES]
        if not cities:
            cities = ["Dehradun", "Delhi", "Mumbai", "Bangalore"]

        temps, humids, winds, valid = [], [], [], []
        for c in cities:
            if c in self.all_weather:
                w = self.all_weather[c]
                temps.append(w["temp"])
                humids.append(w["humidity"])
                winds.append(w["wind_speed"])
                valid.append(c)
        if not valid:
            return

        ax = self.ax_cmp
        ax.clear()
        self._style_ax(ax, "🏙️  City-by-City Comparison")

        x, bw = np.arange(len(valid)), 0.25
        b1 = ax.bar(x - bw, temps,  bw, color=C["accent3"], alpha=0.85, label="Temp (°C)")
        b2 = ax.bar(x,      humids, bw, color=C["accent"],  alpha=0.85, label="Humidity (%)")
        b3 = ax.bar(x + bw, winds,  bw, color=C["accent4"], alpha=0.85, label="Wind (km/h)")

        # Add value labels on top of each bar
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5,
                            color=C["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(valid, rotation=22, ha="right", fontsize=9)
        ax.legend(facecolor=C["card2"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=9)
        self.fig_cmp.tight_layout()
        self.canvas_cmp.draw()

    # ── CHART 5: Analytics Dashboard ─────────────────────────────────────────

    def _draw_analytics(self):
        """
        4-panel analytics dashboard showing the past 7 days of data:
          Top-left:  Humidity over time (area chart)
          Top-right: Wind speed over time (area chart)
          Bot-left:  Weather condition distribution (pie chart)
          Bot-right: Temperature ranking of all cities (horizontal bar)
        """
        if not self.observations or not self.all_weather:
            return

        obs    = self.observations
        times  = [o["datetime"]   for o in obs]
        humids = [o["humidity"]   for o in obs]
        winds  = [o["wind_speed"] for o in obs]

        # Humidity area chart
        ax = self.ax_an1; ax.clear(); self._style_ax(ax, "💧 Humidity (7-day)")
        ax.fill_between(times, humids, alpha=0.30, color=C["accent"])
        ax.plot(times, humids, color=C["accent"], linewidth=1.2)
        ax.set_ylabel("%", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

        # Wind speed area chart
        ax = self.ax_an2; ax.clear(); self._style_ax(ax, "💨 Wind Speed (7-day)")
        ax.fill_between(times, winds, alpha=0.30, color=C["accent4"])
        ax.plot(times, winds, color=C["accent4"], linewidth=1.2)
        ax.set_ylabel("km/h", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

        # Condition distribution pie chart
        ax = self.ax_an3; ax.clear()
        ax.set_facecolor(C["card2"])
        ax.set_title("🌤️  Conditions (All Cities)", color=C["text"], fontsize=9, pad=4)
        conds  = [d["condition"] for d in self.all_weather.values()]
        series = pd.Series(conds).value_counts()
        colors = [C["accent"], C["accent2"], C["accent3"], C["accent4"], C["warning"]]
        wedges, texts, autos = ax.pie(
            series.values, labels=series.index, autopct="%1.0f%%",
            colors=colors[:len(series)],
            textprops={"color": C["text"], "fontsize": 8}, startangle=90)
        for a in autos:
            a.set_color(C["bg"]); a.set_fontsize(8)

        # Temperature ranking (sorted coldest to hottest)
        ax = self.ax_an4; ax.clear(); self._style_ax(ax, "🌡️  Temp Spectrum (All Cities)")
        sorted_d = sorted(self.all_weather.items(), key=lambda x: x[1]["temp"])
        s_cities = [d[0]          for d in sorted_d]
        s_temps  = [d[1]["temp"]  for d in sorted_d]
        vmin, vmax = min(s_temps), max(s_temps)
        bar_colors = plt.cm.RdYlBu_r(
            [(t - vmin) / max(1, vmax - vmin) for t in s_temps])
        ax.barh(s_cities, s_temps, color=bar_colors, alpha=0.88)
        ax.set_xlabel("°C", fontsize=8)
        ax.tick_params(labelsize=7)

        self.fig_an.tight_layout(pad=1.8)
        self.canvas_an.draw()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _on_city_change(self, city: str):
        """Called when user picks a different city from the dropdown."""
        self.city = city
        self._load_async()

    def _status(self, msg: str):
        """Thread-safe status bar update."""
        self.after(0, lambda: self.lbl_status.configure(text=msg))


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Start the app
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
