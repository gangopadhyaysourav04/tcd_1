import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


st.set_page_config(page_title="Toxic Comment Detection", layout="wide")


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()

    mappings = {
        "g@ndu": "gandu", "g@ndoo": "gandu", "g4ndu": "gandu",
        "chutiyaa": "chutiya", "chut!ya": "chutiya",
        "madarch0d": "madarchod", "bhench0d": "bhenchod",
        "lawde": "lauda", "lodu": "lauda",
        "kutter ka baccha": "kutta",
        "bhen ke lode": "bhenchod",
    }

    for k, v in mappings.items():
        text = text.replace(k, v)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- LOAD + FIX DATA ----------------
def load_data():
    df = pd.read_csv("labeled_data_with_more_hinglish_final_v2_cleaned.csv")

    # hard sanitize labels
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class", "tweet"])
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin([0, 1, 2])]

    df["clean_tweet"] = df["tweet"].apply(clean_text)

    # inject neutral
    neutral = [
        "good morning", "thank you", "how are you", "nice work",
        "great job", "namaste", "jai hind", "take care",
        "all good bro", "sab theek hai"
    ]

    df_neutral = pd.DataFrame({
        "class": 2,
        "tweet": neutral
    })
    df_neutral["clean_tweet"] = df_neutral["tweet"].apply(clean_text)

    # inject offensive reinforcement
    reinforce = [
        "madarchod hai tu", "bhenchod sala",
        "lawde ke bacche", "chutiya insaan",
        "teri maa ka bhosda"
    ]

    df_reinforce = pd.DataFrame({
        "class": 1,
        "tweet": reinforce
    })
    df_reinforce["clean_tweet"] = df_reinforce["tweet"].apply(clean_text)

    df = pd.concat([df, df_neutral, df_reinforce], ignore_index=True)

    # enforce minimum samples per class
    counts = df["class"].value_counts()
    for c in [0, 1, 2]:
        if c not in counts or counts[c] < 5:
            raise ValueError(f"Class {c} has insufficient samples: {counts.to_dict()}")

    # balance
    max_n = counts.max()
    df_balanced = pd.concat([
        resample(df[df["class"] == c], replace=True, n_samples=max_n, random_state=42)
        for c in [0, 1, 2]
    ])

    return df_balanced.reset_index(drop=True)


data = load_data()


# ---------------- FEATURES ----------------
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["clean_tweet"])
y = data["class"].values


# final sanity check (this is non-negotiable)
unique, counts = np.unique(y, return_counts=True)
if len(unique) != 3:
    raise ValueError(f"Training labels invalid: {dict(zip(unique, counts))}")


# ---------------- TRAIN MODEL ----------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train collapsed to single class")

    model = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=600
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds,
        labels=[0, 1, 2],
        target_names=["Hate Speech", "Offensive", "Non-toxic"],
        output_dict=True
    )
    cm = confusion_matrix(y_test, preds)

    return model, acc, report, cm


model, acc, report, cm = train_model(X, y)


# ---------------- UI ----------------
st.title("Toxic Comment Detection")

st.sidebar.metric("Accuracy", f"{acc:.2f}")

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Hate", "Offensive", "Non-toxic"],
    yticklabels=["Hate", "Offensive", "Non-toxic"],
    ax=ax
)
st.pyplot(fig)

text = st.text_area("Enter text")

if st.button("Analyze") and text.strip():
    cleaned = clean_text(text)
    X_in = vectorizer.transform([cleaned])
    pred = model.predict(X_in)[0]
    prob = model.predict_proba(X_in)[0][pred]

    labels = {0: "Hate Speech", 1: "Offensive", 2: "Non-toxic"}
    st.success(f"Prediction: {labels[pred]}  |  Confidence: {prob:.2f}")
