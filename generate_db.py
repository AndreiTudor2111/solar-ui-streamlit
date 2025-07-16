# generate_db.py

import os
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import firestore, storage as fb_storage

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

# ─── Firebase helpers ──────────────────────────────────────────────────────────
def init_firebase():
    """
    Retrieve the Firestore client and Storage bucket from the already‐initialized
    Firebase Admin App (initialized by your Streamlit entrypoint).
    """
    if not firebase_admin._apps:
        raise RuntimeError(
            "Firebase Admin has not been initialized. "
            "Make sure init_firebase() in your Streamlit app ran first."
        )
    db = firestore.client()
    bucket = fb_storage.bucket()
    return db, bucket

# ─── Core data pipeline ─────────────────────────────────────────────────────────
def create_user_database(user_id: str,
                         lat: float,
                         lon: float,
                         threshold: int):
    """
    Build the historical DataFrame in hourly resolution,
    engineer features, train an ExtraTrees selector, and
    write three local files into users/{user_id}/:
      - Stan.csv
      - scaler.pkl
      - features.json
    """
    user_dir = os.path.join("users", user_id)
    os.makedirs(user_dir, exist_ok=True)

    # Define year intervals with today() – 3 days
    today = datetime.utcnow().date()
    end_2025 = today - timedelta(days=3)

    intervals = [
        (datetime(2000,1,1).date(), datetime(2009,12,31).date()),
        (datetime(2010,1,1).date(), datetime(2019,12,31).date()),
        (datetime(2020,1,1).date(), datetime(2020,12,31).date()),
        (datetime(2021,1,1).date(), datetime(2021,12,31).date()),
        (datetime(2022,1,1).date(), datetime(2022,12,31).date()),
        (datetime(2023,1,1).date(), datetime(2023,12,31).date()),
        (datetime(2024,1,1).date(), datetime(2024,12,31).date()),
        (datetime(2025,1,1).date(), end_2025)
    ]

    dfs = []
    for start, end in intervals:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            f"&hourly=temperature_2m,cloud_cover,wind_gusts_10m,"
            f"shortwave_radiation,relative_humidity_2m,dew_point_2m"
            f"&timezone=Europe%2FBucharest"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        dfs.append(df)

    df = pd.concat(dfs).sort_index()

    # Feature engineering
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["sunshine_duration"] = (60 * (1 - df["cloud_cover"] / 100)).clip(lower=0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_u"] = 0  # placeholder if you add u/v later
    df["will_charge"] = (df["shortwave_radiation"] > threshold).astype(int)
    df.dropna(inplace=True)

    # Save Stan.csv
    stan_path = os.path.join(user_dir, "Stan.csv")
    df.to_csv(stan_path)

    # Feature selection
    feature_cols = [
        "temperature_2m", "cloud_cover", "wind_gusts_10m",
        "shortwave_radiation", "relative_humidity_2m",
        "dew_point_2m", "sunshine_duration",
        "hour_cos", "hour_sin", "month_cos", "wind_u"
    ]
    X = df[feature_cols]
    y = df["will_charge"]

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected = importances.nlargest(10).index.tolist()

    # Save scaler.pkl
    scaler = StandardScaler().fit(X[selected])
    scaler_path = os.path.join(user_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save features.json
    features_path = os.path.join(user_dir, "features.json")
    pd.Series(selected).to_json(features_path, orient="values")

    return stan_path, scaler_path, features_path

# ─── Public API: generate + upload ──────────────────────────────────────────────
def generate_and_upload_db(
    user_id: str,
    localitate_id: str,
    lat: float,
    lon: float,
    threshold: int
):
    """
    Create the local DB files and then upload them under 
    datasets/{user_id}/{localitate_id}_{timestamp}/ in Cloud Storage,
    and store run metadata in Firestore.
    """
    db, bucket = init_firebase()

    # Firestore: record this run
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_ref = db.collection("models").document(user_id).collection("runs").document(run_id)
    run_ref.set({
        "localitate_id": localitate_id,
        "lat": lat,
        "lon": lon,
        "threshold": threshold,
        "status": "pending",
        "created_at": datetime.utcnow()
    })

    # Create files
    stan_path, scaler_path, features_path = create_user_database(
        user_id, lat, lon, threshold
    )

    # Upload to Storage
    prefix = f"datasets/{user_id}/{run_id}/"
    for local_path in (stan_path, scaler_path, features_path):
        name = os.path.basename(local_path)
        blob = bucket.blob(prefix + name)
        blob.upload_from_filename(local_path)

    # Mark completed
    run_ref.update({"status": "completed", "completed_at": datetime.utcnow()})
    print(f"✅ Run {run_id} generated & uploaded!")



# # generate_db.py
# import os
# import pandas as pd
# import numpy as np
# import joblib
# import requests
# from datetime import datetime, timedelta

# import firebase_admin
# from firebase_admin import credentials, firestore, storage as fb_storage

# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import ExtraTreesClassifier


# def init_firebase(admin_cred_path: str, storage_bucket: str):
#     """
#     Initialize Firebase Admin SDK with the given credentials and storage bucket.

#     Returns:
#         db (firestore.Client): Firestore client
#         bucket (storage.Bucket): Cloud Storage bucket
#     """
#     if not firebase_admin._apps:
#         cred = credentials.Certificate(admin_cred_path)
#         firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
#     db = firestore.client()
#     bucket = fb_storage.bucket()
#     return db, bucket


# def create_user_database(user_id: str, lat: float, lon: float, threshold: int):
#     """
#     Fetch historical hourly weather data for multiple intervals,
#     engineer features, label binary target (charge/no charge) based on threshold,
#     select top features, scale them, and save artifacts locally.

#     Args:
#         user_id (str): Identifier for user, used in local file path
#         lat (float): Latitude of location
#         lon (float): Longitude of location
#         threshold (int): Solar radiation threshold for binary label

#     Returns:
#         stan_path (str): Path to saved CSV dataset
#         scaler_path (str): Path to saved scaler pickle
#         features_path (str): Path to saved selected features JSON
#     """
#     # Prepare user directory
#     user_dir = os.path.join('users', user_id)
#     os.makedirs(user_dir, exist_ok=True)

#     # Define date intervals and adjust 2025 end date to today-3
#     today = datetime.utcnow().date()
#     end_2025 = today - timedelta(days=3)
#     intervals = [
#         ('2000-01-01', '2009-12-31'),
#         ('2010-01-01', '2019-12-31'),
#         ('2020-01-01', '2020-12-31'),
#         ('2021-01-01', '2021-12-31'),
#         ('2022-01-01', '2022-12-31'),
#         ('2023-01-01', '2023-12-31'),
#         ('2024-01-01', '2024-12-31'),
#         ('2025-01-01', end_2025.isoformat())
#     ]

#     # Fetch and concatenate data
#     dfs = []
#     for start, end in intervals:
#         url = (
#             f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
#             f"&start_date={start}&end_date={end}"
#             f"&hourly=temperature_2m,cloud_cover,wind_gusts_10m,shortwave_radiation,"  # noqa
#             f"relative_humidity_2m,dew_point_2m&timezone=Europe%2FBucharest"
#         )
#         resp = requests.get(url)
#         hourly = resp.json().get('hourly', {})
#         df = pd.DataFrame(hourly)
#         df['time'] = pd.to_datetime(df['time'])
#         df.set_index('time', inplace=True)
#         dfs.append(df)
#     df = pd.concat(dfs).sort_index()

#     # Feature engineering
#     df['hour'] = df.index.hour
#     df['month'] = df.index.month
#     # Sunshine duration inversely related to cloud_cover
#     df['sunshine_duration'] = np.maximum(0, 60 * (1 - df['cloud_cover'] / 100))
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
#     df['wind_u'] = 0  # placeholder for future use

#     # Binary label: 1 if radiation > threshold, else 0
#     df['will_charge'] = (df['shortwave_radiation'] > threshold).astype(int)

#     df.dropna(inplace=True)

#     # Save full dataset
#     stan_path = os.path.join(user_dir, 'Stan.csv')
#     df.to_csv(stan_path)

#     # Select features
#     features = [
#         'temperature_2m', 'cloud_cover', 'wind_gusts_10m', 'shortwave_radiation',
#         'relative_humidity_2m', 'dew_point_2m', 'sunshine_duration',
#         'hour_cos', 'hour_sin', 'month_cos', 'wind_u'
#     ]
#     X = df[features]
#     y = df['will_charge']

#     # Train feature importance model
#     model = ExtraTreesClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)
#     importances = pd.Series(model.feature_importances_, index=features)
#     selected = importances.nlargest(10).index.tolist()

#     # Fit and save scaler on selected features
#     scaler = StandardScaler().fit(X[selected])
#     scaler_path = os.path.join(user_dir, 'scaler.pkl')
#     joblib.dump(scaler, scaler_path)

#     # Save selected feature names
#     features_path = os.path.join(user_dir, 'features.json')
#     with open(features_path, 'w') as f:
#         f.write(pd.Series(selected).to_json(orient='values'))

#     return stan_path, scaler_path, features_path


# def generate_and_upload_db(
#     user_id: str,
#     localitate_id: str,
#     lat: float,
#     lon: float,
#     threshold: int,
#     admin_cred_path: str,
#     storage_bucket: str
# ):
#     """
#     Orchestrates the full run: save run config, generate data, upload artifacts, update status.

#     Args:
#         user_id (str): Authenticated user email or ID
#         localitate_id (str): Chosen locality ID from UI
#         lat (float), lon (float): Coordinates
#         threshold (int): Radiation threshold
#         admin_cred_path (str): Path to Firebase admin JSON
#         storage_bucket (str): GCS bucket name
#     """
#     # Initialize Firebase
#     db, bucket = init_firebase(admin_cred_path, storage_bucket)

#     # Create a new run entry
#     run_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')
#     run_ref = db.collection('models').document(user_id).collection('runs').document(run_id)
#     run_ref.set({
#         'localitate_id': localitate_id,
#         'lat': lat,
#         'lon': lon,
#         'threshold': threshold,
#         'created_at': datetime.utcnow(),
#         'status': 'running'
#     })

#     try:
#         # Generate local files
#         stan_path, scaler_path, features_path = create_user_database(
#             user_id, lat, lon, threshold
#         )

#         # Upload each artifact under a run-specific prefix
#         prefix = f"datasets/{user_id}/{run_id}/"
#         for path in [stan_path, scaler_path, features_path]:
#             blob = bucket.blob(prefix + os.path.basename(path))
#             blob.upload_from_filename(path)

#         # Mark run completed
#         run_ref.update({'status': 'completed', 'completed_at': datetime.utcnow()})
#     except Exception as e:
#         run_ref.update({'status': 'error', 'error': str(e)})
#         raise
