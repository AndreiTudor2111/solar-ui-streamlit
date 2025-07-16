# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 21:38:38 2025

@author: ostac
"""


# ui/auth.py
import os
import sys
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Ensure parent folder (project root) is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import setup UI and DB generation
from setup_ui import setup_database_ui
from generate_db import generate_and_upload_db

# Initialize Firebase Admin once
def init_firebase():
    if not firebase_admin._apps:
        cred_path = st.secrets['firebase_admin']['cred_path']
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# Session state defaults
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user_email'] = ''

# --- Authentication Pages ---

def signup_page():
    st.header("Creare cont nou")
    prenume = st.text_input("Prenume", key="signup_prenume")
    email = st.text_input("Email", key="signup_email").strip().lower()
    parola = st.text_input("Parolă", type="password", key="signup_parola")
    parola2 = st.text_input("Confirmă parola", type="password", key="signup_parola2")
    if st.button("Înregistrează", key="signup_btn"):
        if not prenume or not email or not parola:
            st.error("Toate câmpurile sunt obligatorii.")
            return False
        if parola != parola2:
            st.error("Parolele nu coincid.")
            return False
        existing = db.collection('users').where('email', '==', email).get()
        if existing:
            st.error("Email deja înregistrat.")
            return False
        user_doc = {
            'first_name': prenume,
            'email': email,
            'password_hash': generate_password_hash(parola),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        db.collection('users').document().set(user_doc)
        st.success("Cont creat! Accesează tab-ul 'Autentificare' pentru a intra în cont.")
        return True
    return False


def login_page():
    st.header("Autentificare")
    email = st.text_input("Email", key="login_email").strip().lower()
    parola = st.text_input("Parolă", type="password", key="login_password")
    if st.button("Autentifică-te", key="login_btn"):
        if not email or not parola:
            st.error("Completează email și parolă.")
            return False
        user_docs = db.collection('users').where('email', '==', email).get()
        if not user_docs:
            st.error("Utilizator inexistent.")
            return False
        user = user_docs[0].to_dict()
        if not check_password_hash(user.get('password_hash', ''), parola):
            st.error("Parolă incorectă.")
            return False
        # Successful login
        st.success("Autentificare reușită!")
        st.session_state['authenticated'] = True
        st.session_state['user_email'] = email
        return True
    return False

# --- Main UI Flow ---
st.sidebar.title("Meniu")
if not st.session_state['authenticated']:
    page = st.sidebar.radio("Navighează:", ["Autentificare", "Înregistrare"])
    if page == "Autentificare":
        login_page()
    else:
        signup_page()
else:
    st.sidebar.success(f"Bun venit, {st.session_state['user_email']}")
    setup_database_ui()
