"""
Application Streamlit — Prédiction de Sentiment sur des avis de restaurant

Lancement :
    streamlit run app.py

Fichiers requis dans le même dossier que app.py :
    - svm_model.pkl
    - tfidf.pkl
    - tokenizer.pkl
    - lstm_model.h5
"""

import re

import joblib
import numpy as np
import streamlit as st

# ── Config de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction de Sentiment — Restaurant",
    page_icon="🍽️",
    layout="centered",
)

MAX_LEN = 100  # identique à l'entraînement du LSTM

LABEL_MAP = {
    0: ("Négatif", "😠", "#E63946"),
    1: ("Neutre", "😐", "#F4A261"),
    2: ("Positif", "😊", "#2A9D8F"),
}


# ── Téléchargement NLTK (une seule fois, mis en cache) ──────────────
@st.cache_resource
def setup_nltk():
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


# ── Chargement des modèles (mis en cache) ────────────────────────────
@st.cache_resource
def load_artifacts():
    import tensorflow as tf

    svm = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    tokenizer = joblib.load("tokenizer.pkl")
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    return svm, tfidf, tokenizer, lstm_model


def preprocesser_nltk(texte: str, stop_words: set) -> str:
    from nltk.tokenize import word_tokenize

    texte_net = re.sub(r"[^\w\s]", " ", str(texte).lower())
    tokens = word_tokenize(texte_net)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def predire_svm(cleaned_text, svm, tfidf):
    vector = tfidf.transform([cleaned_text])
    prediction = svm.predict(vector)[0]
    scores = svm.decision_function(vector)[0]
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()
    return int(prediction), probs


def predire_lstm(cleaned_text, tokenizer, lstm_model):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = lstm_model.predict(padded, verbose=0)[0]
    prediction = int(np.argmax(probs))
    return prediction, probs


# ── Interface ──────────────────────────────────────────────────────
st.title("🍽️ Prédiction de Sentiment — Avis Restaurant")
st.caption("Entrez un avis en anglais et choisissez un modèle pour prédire son sentiment.")

with st.spinner("Chargement des modèles..."):
    stop_words = setup_nltk()
    try:
        svm, tfidf, tokenizer, lstm_model = load_artifacts()
        artifacts_ok = True
    except Exception as e:
        artifacts_ok = False
        load_error = e

if not artifacts_ok:
    st.error(
        "Impossible de charger les modèles. Vérifiez que les fichiers "
        "`svm_model.pkl`, `tfidf.pkl`, `tokenizer.pkl` et `lstm_model.h5` "
        "se trouvent dans le même dossier que `app.py`."
    )
    st.exception(load_error)
    st.stop()

model_choice = st.radio(
    "Modèle :",
    options=["SVM (TF-IDF)", "LSTM (Deep Learning)"],
    horizontal=True,
)

avis = st.text_area(
    "Avis :",
    placeholder="Entrez votre avis en anglais ici...",
    height=120,
)

if st.button("🔍 Prédire le sentiment", type="primary"):
    avis_clean_input = avis.strip()

    if not avis_clean_input:
        st.warning("⚠️ Veuillez entrer un avis avant de prédire.")
    else:
        cleaned = preprocesser_nltk(avis_clean_input, stop_words)

        with st.expander("📝 Texte nettoyé"):
            st.write(cleaned if cleaned else "*(vide après nettoyage)*")

        if model_choice == "SVM (TF-IDF)":
            prediction, probs = predire_svm(cleaned, svm, tfidf)
            model_used = "SVM"
        else:
            prediction, probs = predire_lstm(cleaned, tokenizer, lstm_model)
            model_used = "LSTM"

        label, emoji, color = LABEL_MAP[prediction]

        st.markdown(f"**Modèle utilisé :** {model_used}")
        st.markdown(
            f"""
            <div style="background-color:{color}22;border:2px solid {color};
                        border-radius:10px;padding:16px;text-align:center;margin:10px 0;">
                <span style="font-size:28px;font-weight:bold;color:{color};">
                    {emoji} Sentiment prédit : {label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Confiance par classe :**")
        for i, (nom, _, c) in LABEL_MAP.items():
            st.write(f"{nom} — {probs[i] * 100:.1f}%")
            st.progress(float(probs[i]))