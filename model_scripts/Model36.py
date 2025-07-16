# Model12_36.py
# Script de antrenare modele LSTM, GRU și Deep GRU pentru orizont de predicție 12–36 ore

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import argparse

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Activează alocarea dinamică a memoriei, ca să nu ocupe tot GPU-ul
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow a detectat GPU: {[gpu.name for gpu in gpus]}")
    except Exception as e:
        print(f"⚠️ Nu am putut activa memory growth pe GPU: {e}")
else:
    print("ℹ️ GPU nu a fost detectat – antrenarea va rula pe CPU.")

def load_and_prepare_sequences(path, target, window=24, shift=180):
    df = pd.read_csv("Stan.csv")
    df = df.rename(columns={"time":"date"})
    df.to_csv("Stan.csv", index=False)

    y = df[target].values
    feature_cols = [col for col in df.columns if col != target]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = []
    for i in range(window, len(df) - shift):
        X.append(df[feature_cols].iloc[i-window:i].values)

    return np.array(X), y[window+shift:]


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_deep_rnn(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        GRU(16),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_model(X, y, model_fn, name):
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = model_fn(X.shape[1:])
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print(f"{name} -> Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
    return model, roc


def main():
    parser = argparse.ArgumentParser(description="Antreneaza modele ML pe 12–36h")
    parser.add_argument('--user_hash', required=True, help='Hash-ul utilizatorului')
    parser.add_argument('--base_dir', default=r"C:\Users\ostac\OneDrive\Desktop\firestore\users", help='Director root')
    parser.add_argument('--window', type=int, default=24)
    parser.add_argument('--shift', type=int, default=180)
    args = parser.parse_args()

    user_dir = os.path.join(args.base_dir, args.user_hash)
    stan_path = os.path.join(user_dir, 'Stan.csv')
    if not os.path.exists(stan_path):
        raise FileNotFoundError(f"Fisierul Stan.csv nu exista pentru {args.user_hash}")

    target = 'will_charge'
    X, y = load_and_prepare_sequences(stan_path, target, window=args.window, shift=args.shift)

    models_info = []
    for fn, name in [(build_lstm, 'LSTM'), (build_gru, 'GRU'), (build_deep_rnn, 'DeepGRU')]:
        model, score = evaluate_model(X, y, fn, name)
        models_info.append((model, score, name))

    best_model, best_score, best_name = max(models_info, key=lambda x: x[1])
    best_path = os.path.join(user_dir, '12_36.keras')
    best_model.save(best_path)
    print(f"Best model: {best_name} with ROC AUC {best_score:.4f} was saved!")


if __name__ == '__main__':
    main()
