import streamlit as st
import numpy as np
import re
import joblib
from nltk.corpus import stopwords
from manual_dt import ManualDecisionTree
from manual_dt import Node

# Load models
multilabel_models = {
    "Decision Tree": 
        joblib.load("multi/m_dt.pkl"),
    "Logistic Regression":
        joblib.load("multi/m_lr.pkl"),
    "Multinomial NB":
        joblib.load("multi/m_mul.pkl"),
    "Random Forest":
        joblib.load("multi/m_rf.pkl")
}

singlelabel_models = {
    "Manual DT": {
        "model": joblib.load("single/s_man_dt.pkl"),
        "type": "bow"
    },
    "Sklearn DT": {
        "model": joblib.load("single/s_sk_dt.pkl"),
        "type": "tfidf"
    },
    "Logistic Regression": {
        "model": joblib.load("single/s_lr.pkl"),
        "type": "tfidf"
    },
    "Bernoulli NB": {
        "model": joblib.load("single/s_ber.pkl"),
        "type": "bow"
    },
    "Multinomial NB": {
        "model": joblib.load("single/s_mul.pkl"),
        "type": "tfidf"
    },
    "Random Forest": {
        "model": joblib.load("single/s_rf.pkl"),
        "type": "tfidf"
    }
}

# Load vocab
vocab = joblib.load("vocab.pkl")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Labels
label_columns = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral'
]

stop_words = set(stopwords.words("english"))

# Preprocessing
def text_split(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words]
    return words

# Vectorization
def vectorize(tokens, vocab):
    vec = np.zeros(len(vocab), dtype=np.uint8)
    for w in tokens:
        if w in vocab:
            vec[vocab[w]] = 1
    return vec

# UI
st.title("Emotion Classification Comparison")
user_input = st.text_area("Enter a comment:")

if st.button("Predict"):
    left_col, right_col = st.columns(2)
    st.subheader("Predictions")
    with left_col:
        # Single-label
        st.subheader("Single-Label Models")
        for name, info in singlelabel_models.items():
            model = info["model"]
            if info["type"] == "bow":
                tokens = text_split(user_input)
                x = vectorize(tokens, vocab)
                x = np.array([x])
            else:
                x = vectorizer.transform([user_input])
            pred = model.predict(x)
            emotion = label_columns[pred[0]]
            st.write(f"**{name}** → {emotion}")
    
    with right_col:
        # Multilabel
        st.subheader("Multi-Label Models")
        for name, model in multilabel_models.items():
            x = vectorizer.transform([user_input])
            # print(f"Name: {name}, Model: {model}")
            pred = model.predict(x)
            probs = model.predict_proba(x)[0]
            probs = model.predict_proba(x)[0]
            threshold = 0.3
            selected = np.where(probs >= threshold)[0]
            if len(selected) == 0:
                selected = np.argsort(probs)[-2:][::-1]
            st.write(f"**{name}**")
            for i in selected:
                st.write(f"{label_columns[i]}: {probs[i]:.3f}")