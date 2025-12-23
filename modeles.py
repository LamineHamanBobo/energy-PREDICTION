# ======================================================
# CHAPITRE 2 – DÉVELOPPEMENT DU MODÈLE LSTM
# ======================================================
# Source des données : sorties de EDA energy2.ipynb
# Environnement : VS Code + virtualenv
# ======================================================


# ================================
# 1. IMPORT DES LIBRAIRIES
# ================================

import argparse
import logging
import os
import sys
import numpy as np
import matplotlib
# backend non-interactif pour exécution headless (serveur / CI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Reproductibilité
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ================================
# 2. CHARGEMENT DES DONNÉES
# ================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees_numeriques")

# Répertoire de base du script et répertoire de sauvegarde des modèles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modeles")
MODEL_FIG_DIR = os.path.join(MODEL_DIR, "figures")

os.makedirs(MODEL_FIG_DIR, exist_ok=True)


def load_npy(path):
    if not os.path.exists(path):
        logging.error("Fichier introuvable: %s", path)
        raise FileNotFoundError(path)
    try:
        return np.load(path)
    except Exception as e:
        logging.exception("Impossible de charger %s: %s", path, e)
        raise


def load_data(data_dir):
    X_train = load_npy(os.path.join(data_dir, "X_train.npy"))
    X_test = load_npy(os.path.join(data_dir, "X_test.npy"))
    y_train = load_npy(os.path.join(data_dir, "y_train.npy"))
    y_test = load_npy(os.path.join(data_dir, "y_test.npy"))
    logging.info("✔ Données chargées")
    logging.info("X_train: %s", X_train.shape)
    logging.info("X_test : %s", X_test.shape)
    return X_train, X_test, y_train, y_test


# ================================
# 3. DIMENSIONS DU PROBLÈME
# ================================

def ensure_3d(X):
    # Si X est (n, timesteps), ajouter canal
    if X.ndim == 2:
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X


def prepare_data(X_train, X_test, y_train, y_test):
    X_train = ensure_3d(X_train)
    X_test = ensure_3d(X_test)
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    return timesteps, n_features, X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    # Scale features across samples and timesteps per feature
    n_samples, timesteps, n_features = X_train.shape
    Xtr_flat = X_train.reshape(-1, n_features)
    Xte_flat = X_test.reshape(-1, n_features)
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_flat)
    Xte_scaled = scaler.transform(Xte_flat)
    X_train_scaled = Xtr_scaled.reshape(n_samples, timesteps, n_features)
    X_test_scaled = Xte_scaled.reshape(X_test.shape[0], timesteps, n_features)
    return X_train_scaled, X_test_scaled, scaler


# ================================
# 4. CONCEPTION DU MODÈLE (2.2)
# ================================

def build_model(timesteps, n_features, units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential(name="LSTM_Energy_Model")
    model.add(
        LSTM(units=units1, return_sequences=True, input_shape=(timesteps, n_features))
    )
    model.add(Dropout(dropout))
    model.add(LSTM(units=units2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


# ================================
# 5. COMPILATION
# ================================

# compilation faite dans build_model


# ================================
# 6. CALLBACKS (2.3)
# ================================

os.makedirs(MODEL_DIR, exist_ok=True)


def get_callbacks(model_dir, patience=10):
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    checkpoint_path = os.path.join(model_dir, "best_lstm_model.keras")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)
    return [early_stopping, checkpoint], checkpoint_path


# ================================
# 7. ENTRAÎNEMENT (2.3)
# ================================

def chronological_train_val_split(X, y, val_frac=0.2):
    n = len(X)
    val_size = max(1, int(n * val_frac))
    if val_size >= n:
        raise ValueError("Dataset trop petit pour la fraction de validation demandée")
    X_tr = X[:-val_size]
    X_val = X[-val_size:]
    y_tr = y[:-val_size]
    y_val = y[-val_size:]
    return X_tr, X_val, y_tr, y_val


def train(model, X_train, y_train, callbacks, epochs=100, batch_size=32):
    X_tr, X_val, y_tr, y_val = chronological_train_val_split(X_train, y_train, val_frac=0.2)
    history = model.fit(
        X_tr,
        y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    return history


# ================================
# 8. COURBES D'APPRENTISSAGE
# ================================



# ================================
# 9. ÉVALUATION SUR JEU DE TEST (2.4)
# ================================

def evaluate_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test).ravel()
    y_true = np.ravel(y_test)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    r2 = r2_score(y_true, y_pred)
    logging.info("=== ÉVALUATION DU MODÈLE ===")
    logging.info("MAE  : %.4f kWh", mae)
    logging.info("RMSE : %.4f kWh", rmse)
    logging.info("MAPE : %.2f %%", mape)
    logging.info("R²   : %.4f", r2)
    return y_pred


def compute_baseline(y_train, y_test):
    """Compute baseline prediction = mean(y_train) and return metrics on y_test."""
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    baseline = float(y_train.mean())
    y_pred_base = np.full_like(y_test, baseline, dtype=float)
    mae = mean_absolute_error(y_test, y_pred_base)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    r2 = r2_score(y_test, y_pred_base)
    logging.info("Baseline (mean train): %.4f", baseline)
    logging.info("Baseline MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)
    return {"baseline": baseline, "mae": mae, "rmse": rmse, "r2": r2}


# ================================
# 10. VISUALISATION PRÉDICTIONS
# ================================

def plot_and_save(y_test, y_pred, model_fig_dir):
    plt.figure()
    plt.plot(np.ravel(y_test), label="Valeurs réelles")
    plt.plot(y_pred, label="Prédictions")
    plt.legend()
    plt.title("Prédictions vs Réalité")
    plt.savefig(os.path.join(model_fig_dir, "predictions_vs_reality.png"), bbox_inches="tight")
    plt.close()


# ================================
# 11. ANALYSE DES RÉSIDUS
# ================================

def save_residuals_plot(y_test, y_pred, model_fig_dir):
    residuals = np.ravel(y_test) - y_pred
    plt.figure()
    plt.plot(residuals)
    plt.axhline(0)
    plt.title("Analyse des résidus")
    plt.savefig(os.path.join(model_fig_dir, "residuals_analysis.png"), bbox_inches="tight")
    plt.close()


def save_model_final(model, model_dir):
    final_path = os.path.join(model_dir, "final_lstm_model.keras")
    try:
        model.save(final_path)
        logging.info("✔ Modèle sauvegardé : %s", final_path)
    except Exception:
        logging.exception("Erreur lors de la sauvegarde finale du modèle")


def main(args):
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    timesteps, n_features, X_train, X_test, y_train, y_test = prepare_data(X_train, X_test, y_train, y_test)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    model = build_model(timesteps, n_features, units1=args.units1, units2=args.units2, dropout=args.dropout, lr=args.lr)
    model.summary()
    callbacks, ckpt = get_callbacks(MODEL_DIR, patience=args.patience)
    history = train(model, X_train, y_train, callbacks, epochs=args.epochs, batch_size=args.batch_size)
    # Sauvegarde des courbes d'apprentissage
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Convergence de la perte (MSE)")
    plt.savefig(os.path.join(MODEL_FIG_DIR, "loss_convergence.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history.get("mae", []), label="Train MAE")
    plt.plot(history.history.get("val_mae", []), label="Val MAE")
    plt.legend()
    plt.title("Évolution de la MAE")
    plt.savefig(os.path.join(MODEL_FIG_DIR, "mae_evolution.png"), bbox_inches="tight")
    plt.close()

    y_pred = evaluate_and_report(model, X_test, y_test)
    plot_and_save(y_test, y_pred, MODEL_FIG_DIR)
    save_residuals_plot(y_test, y_pred, MODEL_FIG_DIR)
    save_model_final(model, MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement LSTM Energy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units1", type=int, default=64)
    parser.add_argument("--units2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logging.exception("Échec de l'exécution: %s", e)



