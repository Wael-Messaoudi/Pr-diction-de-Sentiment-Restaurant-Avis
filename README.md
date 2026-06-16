# Restaurant Review Sentiment Analysis

**Restaurant Review Sentiment Analysis** est une application web développée avec Streamlit permettant de prédire automatiquement le sentiment d’un avis de restaurant en anglais. L’application intègre deux approches complémentaires : un modèle de Machine Learning basé sur **SVM + TF-IDF** et un modèle de Deep Learning basé sur **LSTM**.

L’objectif est de classifier les avis clients en trois catégories :

- 😊 Positif
- 😐 Neutre
- 😠 Négatif

L’interface offre une expérience simple et intuitive permettant de comparer les prédictions des deux modèles en temps réel.

## Fonctionnalités

### Analyse automatique des sentiments
- **Classification multi-classes :** Positif, Neutre et Négatif.
- **Prédiction instantanée** à partir d’un avis saisi par l’utilisateur.
- **Affichage du niveau de confiance** pour chaque classe.
- **Comparaison de modèles :**
  - SVM avec représentation TF-IDF
  - LSTM basé sur les séquences de texte

### Prétraitement NLP
Avant la prédiction, les avis passent par plusieurs étapes de nettoyage :
- Conversion en minuscules.
- Suppression de la ponctuation et des caractères spéciaux.
- Tokenisation avec NLTK.
- Suppression des stop words.
- Suppression des mots très courts.

### Interface utilisateur
- Interface moderne développée avec Streamlit.
- Sélection du modèle via boutons radio.
- Affichage du texte nettoyé.
- Visualisation du sentiment prédit avec couleur et emoji.
- Barres de progression pour représenter les probabilités de chaque classe.

## Pipeline de traitement

Le processus de prédiction suit les étapes suivantes :

1. **Saisie de l’avis utilisateur**
2. **Nettoyage du texte**
3. **Transformation du texte**
   - TF-IDF pour le modèle SVM
   - Tokenizer + Padding pour le modèle LSTM
4. **Prédiction du sentiment**
5. **Affichage du résultat et des probabilités**

## Technologies utilisées

- **Python 3**
- **Streamlit** pour l’interface web
- **Scikit-Learn** pour le modèle SVM
- **TensorFlow / Keras** pour le modèle LSTM
- **NLTK** pour le prétraitement NLP
- **NumPy** pour les calculs numériques
- **Joblib** pour la sauvegarde et le chargement des modèles

## Architecture des modèles

### Modèle SVM

```text
Texte
 ↓
Prétraitement NLP
 ↓
TF-IDF
 ↓
SVM
 ↓
Prédiction
```

### Modèle LSTM

```text
Texte
 ↓
Prétraitement NLP
 ↓
Tokenizer
 ↓
Padding
 ↓
LSTM
 ↓
Softmax
 ↓
Prédiction
```

## Installation et lancement

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/Wael-Messaoudi/Restaurant-Review-Sentiment-Analysis.git

cd Restaurant-Review-Sentiment-Analysis
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

ou

```bash
pip install streamlit tensorflow scikit-learn nltk numpy joblib
```

### Fichiers requis

Les fichiers suivants doivent être placés dans le même dossier que `app.py` :

```text
svm_model.pkl
tfidf.pkl
tokenizer.pkl
lstm_model.h5
```

### Lancement

```bash
streamlit run app.py
```

L’application sera accessible à l’adresse :

```text
http://localhost:8501
```

## Structure du projet

- `app.py` : Interface Streamlit et logique de prédiction.
- `svm_model.pkl` : Modèle SVM entraîné.
- `tfidf.pkl` : Vectoriseur TF-IDF.
- `tokenizer.pkl` : Tokenizer utilisé pour le modèle LSTM.
- `lstm_model.h5` : Modèle LSTM entraîné.
- `requirements.txt` : Dépendances du projet.
- `README.md` : Documentation du projet.

## Résultats

Le système prédit l’une des trois classes suivantes :

| Classe | Description |
|----------|------------|
| 0 | Négatif 😠 |
| 1 | Neutre 😐 |
| 2 | Positif 😊 |

Les probabilités associées à chaque classe sont également affichées afin d’interpréter le niveau de confiance du modèle.

## Améliorations futures

- Support multilingue (Français, Arabe, Anglais).
- Ajout de modèles Transformers (BERT, DistilBERT).
- Déploiement sur Streamlit Cloud.
- Historique des prédictions.
- Analyse d’aspects (service, nourriture, ambiance).
- Comparaison détaillée des performances des modèles.

## Auteur

**Wael Messaoudi**

Étudiant en Intelligence Artificielle, Data Science et Machine Learning.
