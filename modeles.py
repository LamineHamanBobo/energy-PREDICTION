# ======================================================
# CHAPITRE 2 – DÉVELOPPEMENT DU MODÈLE LSTM
# ======================================================
# Source des données : sorties de EDA energy2.ipynb
# Environnement : VS Code + virtualenv
# ======================================================


# ================================
# 1. IMPORT DES LIBRAIRIES
# ================================

import os
import numpy as np
import matplotlib
# backend non-interactif pour exécution headless (serveur / CI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ================================
# 2. CHARGEMENT DES DONNÉES
# ================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees_numeriques")

# Répertoire de base du script et répertoire de sauvegarde des modèles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modeles")
MODEL_FIG_DIR = os.path.join(MODEL_DIR, "figures")

os.makedirs(MODEL_FIG_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("✔ Données chargées")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ================================
# 3. DIMENSIONS DU PROBLÈME
# ================================

timesteps = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# nombre de features (canal) après reshape
n_features = X_train.shape[2]


# ================================
# 4. CONCEPTION DU MODÈLE (2.2)
# ================================

model = Sequential(name="LSTM_Energy_Model")

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(timesteps, 1)
    )
)
model.add(Dropout(0.2))

model.add(
    LSTM(
        units=32,
        return_sequences=False
    )
)
model.add(Dropout(0.2))

model.add(Dense(1))


# ================================
# 5. COMPILATION
# ================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ================================
# 6. CALLBACKS (2.3)
# ================================

os.makedirs(MODEL_DIR, exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_lstm_model.h5"),
    monitor="val_loss",
    save_best_only=True
)


# ================================
# 7. ENTRAÎNEMENT (2.3)
# ================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)


# ================================
# 8. COURBES D'APPRENTISSAGE
# ================================

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Convergence de la perte (MSE)")
plt.savefig(os.path.join(MODEL_FIG_DIR, "loss_convergence.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.legend()
plt.title("Évolution de la MAE")
plt.savefig(os.path.join(MODEL_FIG_DIR, "mae_evolution.png"), bbox_inches="tight")
plt.close()


# ================================
# 9. ÉVALUATION SUR JEU DE TEST (2.4)
# ================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print("\n=== ÉVALUATION DU MODÈLE ===")
print(f"MAE  : {mae:.4f} kWh")
print(f"RMSE : {rmse:.4f} kWh")
print(f"MAPE : {mape:.2f} %")
print(f"R²   : {r2:.4f}")


# ================================
# 10. VISUALISATION PRÉDICTIONS
# ================================

plt.figure()
plt.plot(y_test, label="Valeurs réelles")
plt.plot(y_pred, label="Prédictions")
plt.legend()
plt.title("Prédictions vs Réalité")
plt.savefig(os.path.join(MODEL_FIG_DIR, "predictions_vs_reality.png"), bbox_inches="tight")
plt.close()


# ================================
# 11. ANALYSE DES RÉSIDUS
# ================================

residuals = y_test - y_pred

plt.figure()
plt.plot(residuals)
plt.axhline(0)
plt.title("Analyse des résidus")
plt.savefig(os.path.join(MODEL_FIG_DIR, "residuals_analysis.png"), bbox_inches="tight")
plt.close()


# Sauvegarde finale du modèle entraîné
final_path = os.path.join(MODEL_DIR, "final_lstm_model.keras")
try:
    model.save(final_path)
    print(f"✔ Modèle sauvegardé : {final_path}")
except Exception as e:
    print("Erreur lors de la sauvegarde finale du modèle:", e)



