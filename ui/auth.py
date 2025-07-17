# ui/auth.py
import os
import sys
import streamlit as st
import re
import json
import requests
import subprocess
import time
import threading

import firebase_admin
from firebase_admin import credentials, firestore

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Path către rădăcina proiectului (un nivel mai sus față de ui/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
HF_AUTOTRAIN_ENDPOINT = st.secrets["huggingface"]["autotrain_endpoint"]  # ex: "https://api-inference.huggingface.co/auto-train/v1/jobs"
AUTO_TRAIN_PROJECT    = st.secrets["huggingface"]["project"]

# Import setup UI
from setup_ui import setup_database_ui

# --- Firebase init ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        # calea relativă către cheia JSON, salvată în secrets.toml
        rel = st.secrets["firebase_admin"]["cred_path"]

        bucket_name = st.secrets["firebase_admin"]["bucket"]
        cred_path = os.path.join(PROJECT_ROOT, rel)
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    return firestore.client()

db = init_firebase()

# --- Session state ---
def init_session_state():
    for key, default in {
        'authenticated': False,
        'user_email': '',
        'user_id': '',
        'current_page': 'login',
        'training_job_id': None,
        'training_status': None,
        'last_check_time': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()

# --- Utils ---
def validate_email(email: str) -> bool:
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email) is not None

def validate_password(password: str):
    if len(password) < 6:
        return False, "Parola trebuie să conțină cel puțin 6 caractere"
    return True, ""

def get_user_data(email: str):
    try:
        docs = db.collection('users').where('email', '==', email).get()
        if docs:
            d = docs[0].to_dict(); d['doc_id'] = docs[0].id
            return d
    except Exception as e:
        st.error(f"Eroare DB: {e}")
    return None

def check_existing_models(user_key: str):
    try:
        runs = db.collection("models").document(user_key).collection("runs").stream()
        return [{"id": d.id, **d.to_dict()} for d in runs]
    except Exception as e:
        st.error(f"Eroare la verificare modele: {e}")
        return []

def submit_training_job(user_id: str, run_id: str, config: dict):
    import requests

    payload = {
        "user_id": user_id,
        "run_id": run_id,
        "config": {
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"]
        }
    }

    try:
        resp = requests.post("https://AndreiTdr-aplciatie.hf.space/run/predict", json=payload, timeout=120)
        resp.raise_for_status()
        response_json = resp.json()
        job_output = response_json.get("data", ["<niciun răspuns>"])[0]
        print("Training job response:", response_json)

        st.success("🚀 Job trimis către Hugging Face Space!")
        st.code(job_output)
        return f"{user_id}_{run_id}"
    except Exception as e:
        st.error(f"❌ Eroare la trimitere job în Space: {e}")
        return None


# # --- AutoTrain submit via REST ---

# def submit_training_job(user_id: str, run_id: str, config: dict):
#     from datetime import datetime

#     # Construiește path absolut spre scriptul train.py
#     train_script = os.path.join(PROJECT_ROOT, "train.py")

#     # Creează config temporar într-un fișier JSON
#     temp_config = {
#     "batch_size": config["batch_size"],
#     "epochs": config["epochs"],
#     "learning_rate": config["learning_rate"]}

#     temp_config_path = os.path.join(PROJECT_ROOT, "train_config.json")
#     with open(temp_config_path, "w") as f:
#         json.dump(temp_config, f)

#     # Comandă: python train.py --user_id ... --run_id ...
#     cmd = [
#         sys.executable,
#         train_script,
#         "--user_id", user_id,
#         "--run_id", run_id,
#         "--project", AUTO_TRAIN_PROJECT
#     ]

#     # Rulează scriptul în fundal
#     try:
#         subprocess.Popen(cmd)
#         st.success("🚀 Job AutoTrain lansat local!")
#         # Salvează și în Firestore că a fost lansat
#         db.collection("training_jobs").document(f"{user_id}_{run_id}").set({
#             "user_id": user_id,
#             "run_id": run_id,
#             "project": AUTO_TRAIN_PROJECT,
#             "status": "submitted",
#             "created_at": datetime.utcnow()
#         })
#         return f"{user_id}_{run_id}"
#     except Exception as e:
#         st.error(f"❌ Eroare la lansarea locală AutoTrain: {e}")
#         return None


#     from datetime import datetime

#     hf_token    = st.secrets["huggingface"]["token"]
#     hf_username = st.secrets["huggingface"]["username"]
#     safe_user   = user_id.replace("@","-").replace(".","-")
#     repo_id     = f"{hf_username}/{safe_user}-{run_id}-dataset"

#     payload = {
#       "dataset":       repo_id,
#       "task":          "tabular-classification",
#       "column_mapping":{
#         "target":              "will_charge",
#         "continuous_features": config.get("feature_columns", [])
#       },
#       "optim_args":    {
#         "num_train_epochs":   config["epochs"],
#         "batch_size":         config["batch_size"]
#       },
#       "model_config":  {
#         "model_size":         "small",
#         "learning_rate":      config["learning_rate"]
#       }
#     }

#     headers = {
#       "Authorization": f"Bearer {hf_token}",
#       "Content-Type":  "application/json"
#     }

#     url = ("https://api.autotrain.huggingface.co/v1/jobs"
# )

#     # 4) POST către Hugging Face AutoTrain
#     try:
#         resp = requests.post(url, headers=headers, json=payload, timeout=60)
#         resp.raise_for_status()
#     except Exception as e:
#         st.error(f"❌ Eroare la lansarea job-ului AutoTrain: {e}")
#         return None

#     job_id = resp.json().get("job_id") or resp.json().get("id")

#     # 5) Salvează în Firestore
#     db.collection("training_jobs").document(job_id).set({
#         "user_id":    user_id,
#         "run_id":     run_id,
#         "project":    "autotrain-advanced",
#         "status":     "submitted",
#         "created_at": datetime.utcnow()
#     })

#     st.success(f"🚀 Job lansat: {job_id}")
#     return job_id

def check_training_status(job_id: str):
    try:
        doc = db.collection("training_jobs").document(job_id).get()
        if not doc.exists:
            return "not_found"
        return doc.to_dict().get("status","unknown")
    except Exception as e:
        st.error(f"Eroare status antrenare: {e}")
        return "error"

# --- Auth forms ---
def signup_page():
    st.header("🔐 Creare cont nou")
    with st.form("signup_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        prenume = col1.text_input("Prenume *")
        email   = col1.text_input("Email *")
        nume    = col2.text_input("Nume")
        telefon = col2.text_input("Telefon")
        p1 = st.text_input("Parolă *", type="password")
        p2 = st.text_input("Confirmă parola *", type="password")
        if st.form_submit_button("🚀 Înregistrează"):
            if not prenume or not email or not p1:
                st.error("Completați câmpurile obligatorii"); return
            if not validate_email(email): 
                st.error("Email invalid"); return
            valid, err = validate_password(p1)
            if not valid:
                st.error(err); return
            if p1!=p2:
                st.error("Parolele nu coincid"); return
            if get_user_data(email):
                st.error("Email deja înregistrat"); return
            # creează
            db.collection("users").document().set({
                "first_name": prenume,
                "last_name":  nume,
                "email":      email,
                "phone":      telefon,
                "password_hash": generate_password_hash(p1),
                "created_at": datetime.utcnow(),
                "is_active": True
            })
            st.success("Cont creat!"); time.sleep(1); st.session_state.current_page="login"; st.rerun()

def login_page():
    st.header("🔑 Autentificare")
    with st.form("login_form"):
        email = st.text_input("Email")
        pwd   = st.text_input("Parolă", type="password")
        if st.form_submit_button("🔐 Autentifică-te"):
            if not email or not pwd:
                st.error("Toate câmpurile"); return
            user = get_user_data(email)
            if not user or not check_password_hash(user["password_hash"], pwd):
                st.error("Email sau parolă invalide"); return
            if not user.get("is_active",True):
                st.error("Cont dezactivat"); return
            st.session_state.authenticated=True
            st.session_state.user_email=email
            st.session_state.user_id=user["doc_id"]
            db.collection("users").document(user["doc_id"]).update({"last_login":datetime.utcnow()})
            st.success("Autentificat!"); time.sleep(1); st.rerun()

def logout():
    for k in ["authenticated","user_email","user_id","training_job_id","training_status"]:
        st.session_state.pop(k,None)
    init_session_state()
    st.rerun()

# --- Training UI (unchanged) ---
def training_interface():
    st.header("🤖 Antrenare Modele")
    user = st.session_state.user_email
    runs = check_existing_models(user)
    # filtrează doar `status=="completed"`
    options, ids = [], []
    for r in runs:
        if r.get("status")=="completed":
            dt = r.get("created_at")
            label = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt,datetime) else r["id"]
            options.append(f"Run {label} ({r['id']})")
            ids.append(r["id"])
    if not options:
        st.warning("⚠️ Generează mai întâi baza de date."); return
    idx = st.selectbox("Alege run:", list(range(len(options))), format_func=lambda i: options[i])
    run_id = ids[idx]
    # configurare
    st.subheader("⚙️ Configurare Antrenare")
    c1,c2 = st.columns(2)
    horizons = c1.multiselect("Orizonturi:", ["12h","36h","7_days"],["12h"])
    batch    = c1.slider("Batch size:",16,128,32)
    epochs   = c2.slider("Epoci:",10,200,50)
    lr       = c2.select_slider("Learning rate:",[1e-4,1e-3,1e-2,1e-1],1e-3)
    models   = st.multiselect("Modele:",["LSTM","GRU","Deep_GRU"],["GRU"])
    if st.button("🚀 Începe Antrenarea"):
        if not horizons or not models:
            st.error("Selectați cel puțin un orizont și un model"); return
        # preia coloanele (din Stan.csv) direct via Firestore sau hardcode
        feature_columns = st.session_state.get("feature_columns", [])
        cfg = {
          "horizons": horizons, "models": models,
          "batch_size": batch, "epochs": epochs,
          "learning_rate": lr,
          "feature_columns": feature_columns
        }
        jid = submit_training_job(user, run_id, cfg)
        if jid:
            st.session_state.training_job_id=jid
            st.session_state.training_status="submitted"
    # status
    if st.session_state.get("training_job_id"):
        st.subheader("📊 Status Antrenare")
        st.info(f"🔄 {check_training_status(st.session_state.training_job_id)}")

# --- Main flow ---
def main_app():
    st.sidebar.title("🌞 Solar Prediction App")
    if user:=get_user_data(st.session_state.user_email):
        st.sidebar.success(f"Bun venit, {user['first_name']}!")
    page = st.sidebar.radio("Navigare:",["🔧 Configurare","🤖 Antrenare","📊 Predicții","⚙️ Setări"])
    if st.sidebar.button("🚪 Logout"): logout()
    if page=="🔧 Configurare":
        setup_database_ui()
    elif page=="🤖 Antrenare":
        training_interface()
    else:
        st.header(page); st.info("În curs de implementare...")

if __name__=="__main__":
    st.set_page_config(page_title="Solar Prediction App",page_icon="🌞",layout="wide")
    if not st.session_state.authenticated:
        st.title("🌞 Solar Prediction App"); st.markdown("---")
        t1,t2 = st.tabs(["🔑 Autentificare","📝 Înregistrare"])
        with t1: login_page()
        with t2: signup_page()
    else:
        main_app()


# # ui/auth.py
# import os
# import sys
# import streamlit as st
# import json
# import firebase_admin
# from firebase_admin import credentials, firestore
# from werkzeug.security import generate_password_hash, check_password_hash
# from datetime import datetime
# import requests
# import json
# import time
# import threading
# import subprocess
# import os

# # Path către rădăcina proiectului (un nivel mai sus față de ui/)
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(PROJECT_ROOT)


# # Import setup UI and DB generation
# from setup_ui import setup_database_ui

# # Initialize Firebase Admin once
# @st.cache_resource
# def init_firebase():
#     if not firebase_admin._apps:
#         rel = st.secrets["firebase_admin"]["cred_path"]
#         bucket_name = st.secrets["firebase_admin"]["bucket"]
#         cred_path = os.path.join(os.getcwd(), rel)
#         cred = credentials.Certificate(cred_path)
#         firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
#     return firestore.client()

# db = init_firebase()


# # Session state initialization
# def init_session_state():
#     """Initialize all session state variables"""
#     defaults = {
#         'authenticated': False,
#         'user_email': '',
#         'user_id': '',
#         'current_page': 'login',
#         'training_job_id': None,
#         'training_status': None,
#         'last_check_time': None
#     }
    
#     for key, default_value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = default_value

# init_session_state()

# # --- Utility Functions ---

# def validate_email(email):
#     """Basic email validation"""
#     import re
#     pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#     return re.match(pattern, email) is not None

# def validate_password(password):
#     """Password validation - minimum requirements"""
#     if len(password) < 6:
#         return False, "Parola trebuie să conțină cel puțin 6 caractere"
#     return True, ""

# def get_user_data(email):
#     """Retrieve user data from Firestore"""
#     try:
#         user_docs = db.collection('users').where('email', '==', email).get()
#         if user_docs:
#             user_doc = user_docs[0]
#             user_data = user_doc.to_dict()
#             user_data['doc_id'] = user_doc.id
#             return user_data
#         return None
#     except Exception as e:
#         st.error(f"Eroare la accesarea bazei de date: {e}")
#         return None

# def check_existing_models(user_key: str):
#     try:
#         runs = db.collection("models").document(user_key).collection("runs").stream()
#         return [{"id": doc.id, **doc.to_dict()} for doc in runs]
#     except Exception as e:
#         st.error(f"Eroare la verificare modele: {e}")
#         return []

# # --- Training job launcher ---
# def submit_training_job(user_id, run_id, config):
#     cmd = [sys.executable, os.path.join(PROJECT_ROOT,"train.py"),
#            "--user_id", user_id,
#            "--run_id",  run_id,
#            "--batch_size", str(config["batch_size"]),
#            "--epochs",     str(config["epochs"]),
#            "--lr",         str(config["learning_rate"])]
#     proc = subprocess.Popen(cmd)
#     st.success(f"🌐 Deschidere Colab (PID {proc.pid})")
#     return proc.pid


# def check_training_status(job_id):
#     try:
#         doc = db.collection("training_jobs").document(str(job_id)).get()
#         if not doc.exists:
#             return "not_found"
#         status = doc.to_dict().get("status", "unknown")
#         return status
#     except Exception as e:
#         st.error(f"Eroare status antrenare: {e}")
#         return "error"

# # --- Authentication Pages ---

# def signup_page():
#     """Enhanced signup page with better validation"""
#     st.header("🔐 Creare cont nou")
    
#     with st.form("signup_form", clear_on_submit=True):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             prenume = st.text_input("Prenume *", placeholder="Introduceți prenumele")
#             email = st.text_input("Email *", placeholder="emailultau@email.com")
        
#         with col2:
#             nume = st.text_input("Nume", placeholder="Introduceți numele")
#             telefon = st.text_input("Telefon", placeholder="07xxxxxxxx")
        
#         parola = st.text_input("Parolă *", type="password", placeholder="Minim 6 caractere")
#         parola2 = st.text_input("Confirmă parola *", type="password")
        
#         submit_button = st.form_submit_button("🚀 Înregistrează", use_container_width=True)
        
#         if submit_button:
#             # Validation
#             if not prenume or not email or not parola:
#                 st.error("Câmpurile marcate cu * sunt obligatorii")
#                 return False
            
#             if not validate_email(email.strip().lower()):
#                 st.error("Format email invalid")
#                 return False
            
#             valid_password, password_error = validate_password(parola)
#             if not valid_password:
#                 st.error(password_error)
#                 return False
            
#             if parola != parola2:
#                 st.error("Parolele nu coincid")
#                 return False
            
#             # Check if user already exists
#             email_clean = email.strip().lower()
#             if get_user_data(email_clean):
#                 st.error("Email-ul este deja înregistrat")
#                 return False
            
#             # Create user
#             try:
#                 user_doc = {
#                     'first_name': prenume,
#                     'last_name': nume or '',
#                     'email': email_clean,
#                     'phone': telefon or '',
#                     'password_hash': generate_password_hash(parola),
#                     'created_at': datetime.utcnow(),
#                     'updated_at': datetime.utcnow(),
#                     'is_active': True
#                 }
                
#                 db.collection('users').document().set(user_doc)
#                 st.success("✅ Cont creat cu succes! Vă puteți autentifica acum.")
#                 time.sleep(2)
#                 st.session_state.current_page = 'login'
#                 st.rerun()
                
#             except Exception as e:
#                 st.error(f"Eroare la crearea contului: {e}")
#                 return False
    
#     return False

# def login_page():
#     """Enhanced login page"""
#     st.header("🔑 Autentificare")
    
#     with st.form("login_form"):
#         email = st.text_input("Email", placeholder="exemplu@email.com")
#         parola = st.text_input("Parolă", type="password", placeholder="Introduceți parola")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             login_button = st.form_submit_button("🔐 Autentifică-te", use_container_width=True)
#         with col2:
#             if st.form_submit_button("📝 Cont nou", use_container_width=True):
#                 st.session_state.current_page = 'signup'
#                 st.rerun()
        
#         if login_button:
#             if not email or not parola:
#                 st.error("Completați toate câmpurile")
#                 return False
            
#             email_clean = email.strip().lower()
#             user_data = get_user_data(email_clean)
            
#             if not user_data:
#                 st.error("Utilizator inexistent")
#                 return False
            
#             if not check_password_hash(user_data.get('password_hash', ''), parola):
#                 st.error("Parolă incorectă")
#                 return False
            
#             if not user_data.get('is_active', True):
#                 st.error("Cont dezactivat")
#                 return False
            
#             # Successful login
#             st.success("✅ Autentificare reușită!")
#             st.session_state.authenticated = True
#             st.session_state.user_email = email_clean
#             st.session_state.user_id = user_data['doc_id']
            
#             # Update last login
#             db.collection('users').document(user_data['doc_id']).update({
#                 'last_login': datetime.utcnow()
#             })
            
#             time.sleep(1)
#             st.rerun()
            
#     return False

# def logout():
#     """Logout functionality"""
#     for key in ['authenticated', 'user_email', 'user_id', 'training_job_id', 'training_status']:
#         if key in st.session_state:
#             del st.session_state[key]
#     init_session_state()
#     st.rerun()

# # --- Training Management Interface ---
# def training_interface():
#     """Interface for model training management"""
#     st.header("🤖 Antrenare Modele")

#     # 1. Fetch all runs under the user's email
#     user_key = st.session_state["user_email"]
#     runs = check_existing_models(user_key)

#     # 2. Filter only completed runs and prepare dropdown options
#     options, run_ids = [], []
#     for run in runs:
#         if run.get("status") == "completed":
#             created = run.get("created_at")
#             if isinstance(created, datetime):
#                 label = created.strftime("%Y-%m-%d %H:%M")
#             else:
#                 label = run["id"]
#             options.append(f"Run {label} ({run['id']})")
#             run_ids.append(run["id"])

#     # 3. If none, warn and exit
#     if not options:
#         st.warning(
#             "⚠️ Nu există date generate. "
#             "Generează mai întâi baza de date în secțiunea de configurare."
#         )
#         return

#     # 4. Select which run to train by index
#     idx = st.selectbox(
#         "Selectați run-ul pentru antrenare:",
#         list(range(len(options))),
#         format_func=lambda i: options[i]
#     )
#     selected_run_id = run_ids[idx]

#     # 5. Training configuration inputs
#     st.subheader("⚙️ Configurare Antrenare")
#     col1, col2 = st.columns(2)
#     with col1:
#         horizons = st.multiselect(
#             "Orizonturi de predicție:",
#             ["12h", "36h", "7_days"],
#             default=["12h"]
#         )
#         batch_size = st.slider("Batch Size:", 16, 128, 30)
#     with col2:
#         epochs = st.slider("Numărul de epoci:", 10, 200, 30)
#         learning_rate = st.select_slider(
#             "Learning Rate:",
#             options=[0.0001, 0.001, 0.01, 0.1],
#             value=0.001
#         )

#     models_to_train = st.multiselect(
#         "Modele de antrenat:",
#         ["LSTM", "GRU", "Deep_GRU"],
#         default=["GRU"]
#     )

#     # 6. Launch training when button is clicked
#     if st.button("🚀 Începe Antrenarea", use_container_width=True):
#         if not horizons or not models_to_train:
#             st.error("Selectați cel puțin un orizont și un model")
#             return

#         model_config = {
#             "horizons": horizons,
#             "models": models_to_train,
#             "batch_size": batch_size,
#             "epochs": epochs,
#             "learning_rate": learning_rate
#         }
#         pid = submit_training_job(user_key, selected_run_id, model_config)
#         if pid:
#             st.session_state["training_job_id"] = pid
#             st.session_state["training_status"] = "submitted"

#     # 7. Show training status if a job is running
#     if st.session_state.get("training_job_id"):
#         status = check_training_status(st.session_state["training_job_id"])
#         st.subheader("📊 Status Antrenare")
#         if status == "completed":
#             st.success("✅ Antrenarea s-a finalizat cu succes!")
#             st.balloons()
#         elif status in ["queued", "training"]:
#             st.info(f"🔄 Stare: {status}…")
#         else:
#             st.error(f"❌ Status eroare: {status}")


# # --- Main Application Flow ---

# def main_app():
#     """Main application interface for authenticated users"""
#     st.sidebar.title("🌞 Solar Prediction App")
    
#     # User info
#     user_data = get_user_data(st.session_state.user_email)
#     if user_data:
#         st.sidebar.success(f"Bun venit, {user_data.get('first_name', 'User')}!")
    
#     # Navigation
#     pages = {
#         "🔧 Configurare Bază": "setup",
#         "🤖 Antrenare Modele": "training",
#         "📊 Predicții": "predictions",
#         "⚙️ Setări": "settings"
#     }
    
#     selected_page = st.sidebar.radio("Navigare:", list(pages.keys()))
    
#     # Logout button
#     if st.sidebar.button("🚪 Ieșire din cont", use_container_width=True):
#         logout()
    
#     # Page routing
#     page_key = pages[selected_page]
    
#     if page_key == "setup":
#         setup_database_ui()
#     elif page_key == "training":
#         training_interface()
#     elif page_key == "predictions":
#         st.header("📊 Predicții")
#         st.info("Funcționalitate în dezvoltare...")
#     elif page_key == "settings":
#         st.header("⚙️ Setări")
#         st.info("Funcționalitate în dezvoltare...")

# # --- Main Entry Point ---

# if __name__ == "__main__":
#     st.set_page_config(
#         page_title="Solar Prediction App",
#         page_icon="🌞",
#         layout="wide"
#     )
    
#     # Main application logic
#     if not st.session_state.authenticated:
#         st.title("🌞 Solar Prediction App")
#         st.markdown("---")
        
#         # Authentication tabs
#         tab1, tab2 = st.tabs(["🔑 Autentificare", "📝 Înregistrare"])
        
#         with tab1:
#             login_page()
        
#         with tab2:
#             signup_page()
#     else:
#         main_app()