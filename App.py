# ======================================================
# APPLICATION STREAMLIT – PRÉVISION ÉNERGÉTIQUE (V2)
# ======================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
 


# ================================
# CONFIGURATION
# ================================

st.set_page_config(
    page_title="Prévision Énergétique – LSTM",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #0d6efd;}
    h2 {color: #198754;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔋 Système Intelligent de Prévision Énergétique")


# ================================
# CHARGEMENT DU MODÈLE
# ================================


@st.cache_resource
def load_lstm_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modeles", "final_lstm_model.keras")
    if not os.path.exists(model_path):
        st.error(f"Modèle introuvable : {model_path}")
        return None
    return load_model(model_path, compile=False)


model = load_lstm_model()
if model is None:
    st.stop()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# ================================
# ONGLET PRINCIPAL
# ================================

tabs = st.tabs([
    "📤 Données",
    "📈 Prédictions",
    "🚦 États & Recommandations",
    "📥 Export"
])


# ================================
# ONGLET 1 : DONNÉES
# ================================

with tabs[0]:
    st.header("Importation des données")

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV / Excel / TXT",
        type=["csv", "xlsx", "xls", "txt"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, delimiter="\t")
            else:
                df = pd.read_excel(uploaded_file)

            st.success("Fichier chargé avec succès")
            st.dataframe(df.head())

            if df.shape[0] < 50:
                st.warning("Le dataset est très petit. Résultats à interpréter avec prudence.")

        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            st.stop()


# ================================
# ONGLET 2 : PRÉDICTIONS
# ================================

with tabs[1]:
    st.header("Prévisions multi-horizons")

    if uploaded_file is None:
        st.info("Veuillez d’abord importer un fichier.")
        st.stop()

    # Sélection du périmètre
    analysis_type = st.radio(
        "Type d’analyse",
        ["Consommation globale", "Par machine"]
    )

    if analysis_type == "Par machine":
        # Filtrer colonnes numériques seulement
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("Aucune colonne numérique trouvée.")
            st.stop()
        machine = st.selectbox("Choisir la machine", numeric_cols)
        energy_series = df[machine].astype(float).values
    else:
        # Sommer les colonnes numériques seulement
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.error("Aucune colonne numérique trouvée.")
            st.stop()
        energy_series = numeric_df.sum(axis=1).astype(float).values

    # Préparation séquence
    FEATURES = 1

    # Déduire TIMESTEPS depuis la forme d'entrée du modèle (ex: (None, 56, 1))
    try:
        TIMESTEPS = int(model.input_shape[1])
    except Exception:
        TIMESTEPS = 24  # valeur de secours

    if len(energy_series) < TIMESTEPS:
        st.error(f"La série est trop courte ({len(energy_series)} < {TIMESTEPS}).")
        st.stop()

    last_sequence = energy_series[-TIMESTEPS:].reshape(1, TIMESTEPS, FEATURES)

    # Horizon
    horizons = {
        "15 minutes": 1,
        "1 heure": 4,
        "6 heures": 24,
        "24 heures": 96
    }

    horizon_choice = st.selectbox("Horizon de prédiction", horizons.keys())
    steps = horizons[horizon_choice]

    # Prévision récursive
    def forecast(model, seq, steps):
        preds = []
        current = seq.copy()
        for _ in range(steps):
            p = float(model.predict(current, verbose=0).ravel()[0])
            preds.append(p)
            # shift along the timestep axis (axis=1)
            current = np.roll(current, -1, axis=1)
            current[0, -1, 0] = p
        return np.array(preds)

    predictions = forecast(model, last_sequence, steps)

    # Intervalles de confiance
    residual_std = np.std(np.diff(energy_series))
    upper = predictions + 1.96 * residual_std
    lower = predictions - 1.96 * residual_std

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions, label="Prévision")
    ax.fill_between(range(len(predictions)), lower, upper, alpha=0.3, label="Intervalle 95%")
    ax.legend()
    ax.set_title("Prévision avec intervalle de confiance")

    st.pyplot(fig)


# ================================
# ONGLET 3 : ÉTATS & RECOMMANDATIONS
# ================================

with tabs[2]:
    st.header("Analyse énergétique")

    q25, q75 = np.percentile(energy_series, [25, 75])

    def state(v):
        if v < q25:
            return "Faible 🟢"
        elif v < q75:
            return "Normale 🟡"
        else:
            return "Critique 🔴"

    states = [state(v) for v in predictions]

    result_df = pd.DataFrame({
        "Prévision (kWh)": predictions,
        "État": states
    })

    st.dataframe(result_df)

    if "Critique 🔴" in states:
        st.error("Consommation critique : réduire les charges non essentielles.")
    elif "Normale 🟡" in states:
        st.warning("Surveillance recommandée.")
    else:
        st.success("Consommation maîtrisée.")


# ================================
# ONGLET 4 : EXPORT
# ================================

with tabs[3]:
    st.header("Téléchargement des résultats")

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Télécharger les résultats (CSV)",
        csv,
        "resultats_prediction.csv",
        "text/csv"
    )
