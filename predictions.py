# ======================================================
# CHAPITRE 2.5 – PRÉDICTIONS MULTI-HORIZONS
# ======================================================

import numpy as np
import matplotlib.pyplot as plt


# ================================
# 1. PARAMÈTRES MULTI-HORIZONS
# ================================

# Horizons exprimés en nombre de pas temporels
# À adapter selon ta granularité (minutes / heures)
HORIZONS = {
    "15_min": 1,
    "1_heure": 4,
    "6_heures": 24,
    "24_heures": 96
}


# ================================
# 2. FONCTION DE PRÉDICTION RÉCURSIVE
# ================================

def recursive_forecast(model, last_sequence, n_steps):
    """
    Prédiction récursive multi-horizons
    
    model : modèle LSTM entraîné
    last_sequence : dernière séquence connue (timesteps, features)
    n_steps : nombre de pas à prédire
    """
    
    sequence = last_sequence.copy()
    predictions = []

    for _ in range(n_steps):
        pred = model.predict(sequence[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)

        # Décalage de la fenêtre temporelle
        new_step = sequence[-1].copy()
        new_step[0] = pred  # mise à jour energy_kWh

        sequence = np.vstack([sequence[1:], new_step])

    return np.array(predictions)


# ================================
# 3. SÉLECTION D’UNE SÉQUENCE TEST
# ================================

# Dernière séquence du jeu de test
last_sequence = X_test[-1]


# ================================
# 4. PRÉDICTIONS MULTI-HORIZONS
# ================================

multi_horizon_predictions = {}

for label, steps in HORIZONS.items():
    preds = recursive_forecast(model, last_sequence, steps)
    multi_horizon_predictions[label] = preds
    print(f"✔ Prédictions générées pour {label} ({steps} pas)")


# ================================
# 5. VISUALISATION DES HORIZONS
# ================================

plt.figure(figsize=(10, 5))

for label, preds in multi_horizon_predictions.items():
    plt.plot(preds, label=label)

plt.title("Prédictions multi-horizons de la consommation énergétique")
plt.xlabel("Pas temporels futurs")
plt.ylabel("Consommation (kWh)")
plt.legend()
plt.show()


# ================================
# 6. SAUVEGARDE DES PRÉDICTIONS
# ================================

os.makedirs("predictions", exist_ok=True)

for label, preds in multi_horizon_predictions.items():
    np.save(f"predictions/predictions_{label}.npy", preds)

print("✔ Toutes les prédictions multi-horizons ont été sauvegardées")
