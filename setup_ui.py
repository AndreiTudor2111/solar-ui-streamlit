# setup_ui.py
import os
import streamlit as st
import pandas as pd
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage as fb_storage
from generate_db import generate_and_upload_db
import json
import tempfile

@st.cache_resource
def init_db():
    if not firebase_admin._apps:
        # ✅ Citește din secrets
        firebase_json = json.loads(st.secrets["firebase_admin"]["firebase_admin_config"])


        # ✅ Scrie configul temporar
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            json.dump(firebase_json, tmp)
            tmp.flush()
            cred = credentials.Certificate(tmp.name)

        bucket_name = st.secrets["firebase_admin"]["bucket"]
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    db = firestore.client()
    bucket = fb_storage.bucket()
    return db, bucket

db, bucket = init_db()

# === UI function for database setup ===
def setup_database_ui():
    st.title("Configurare bază istorică Open-Meteo")

    user_id = st.session_state["user_email"]
    runs_ref = db.collection("models").document(user_id).collection("runs")
    existing = [doc.id for doc in runs_ref.stream()]

    # Offer deletion of old runs
    if existing:
        st.subheader("Modele existente")
        for rid in existing:
            st.write(f"• {rid}")
        if st.button("Șterge toate modelele existente"):
            for rid in existing:
                runs_ref.document(rid).delete()
            st.success("Modelele existente au fost șterse.")
            existing = []

    st.markdown("---")
    st.markdown("### Noua configurare")

    @st.cache_data
    def load_localities():
        #base_dir = os.path.dirname(__file__)
        path = os.path.join(os.getcwd(), "ui", "localitati.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nu găsesc localitati.csv la {path}")
        return pd.read_csv(path)

    df = load_localities()
    selected_name = st.selectbox("Localitate:", sorted(df["nume"].unique()))
    row = df[df["nume"] == selected_name].iloc[0]
    localitate_id = str(int(row["id"]))
    lat, lon = float(row["lat"]), float(row["lng"])
    threshold = st.slider("Prag radiație solară (W/m²)", 0, 400, 150, 10)

    if st.button("Generează nouă bază și salvează model"):
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        runs_ref.document(run_id).set({
            "localitate_id": localitate_id,
            "lat": lat,
            "lon": lon,
            "threshold": threshold,
            "status": "pending",
            "created_at": datetime.utcnow()
        })
        with st.spinner("Se generează și se încarcă baza de date…"):
            try:
                # Kick off the backend job
                generate_and_upload_db(
                    user_id=user_id,
                    localitate_id=localitate_id,
                    lat=lat,
                    lon=lon,
                    threshold=threshold,
                    admin_cred_path=None,        # not used anymore
                    storage_bucket=None         # not used anymore
                )
                runs_ref.document(run_id).update({"status": "completed"})
                st.success(f"Run {run_id} generat cu succes!")
            except Exception as e:
                runs_ref.document(run_id).update({"status": "error", "error": str(e)})
                st.error(f"Eroare la generare: {e}")
