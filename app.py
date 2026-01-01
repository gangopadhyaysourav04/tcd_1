import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


st.set_page_config(page_title="Toxic Comment Detection", page_icon="⚠️")


def clean_text(text):
    text = str(text).lower()

    mappings = {
        "g@ndu": "gandu", "g@ndoo": "gandu", "g4ndu": "gandu",
        "chutiyaa": "chutiya", "chut!ya": "chutiya", "chutiy@": "chutiya",
        "madarch0d": "madarchod", "bhench0d": "bhenchod",
        "fck": "fuck", "fucc": "fuck",
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


def load_data():
    df = pd.read_csv("labeled_data_with_more_hinglish_final_v2_cleaned.csv")
    df = df[["class", "tweet"]].dropna()
    df["clean_tweet"] = df["tweet"].apply(clean_text)

    neutral_comments = [
        "good morning", "good night", "thank you", "thanks bhai",
        "how are you", "take care", "nice work", "great job",
        "jai hind", "namaste", "radhe radhe", "shukriya",
        "salaam alaikum", "ram ram", "sat sri akaal",
        "namaskar", "kaise ho", "kya haal hai",
        "sab theek hai", "all good bro"
    ]

    df_neutral = pd.DataFrame({
        "class": [2] * len(neutral_comments),
        "tweet": neutral_comments
    })
    df_neutral["clean_tweet"] = df_neutral["tweet"].apply(clean_text)

    reinforcement = [
        "madarchod hai tu", "lawde ke bacche",
        "bhenchod sala", "chutiya insaan",
        "teri maa ka bhosda", "gandu aadmi"
    ]

    df_reinforce = pd.DataFrame({
        "class": [1] * len(reinforcement),
        "tweet": reinforcement
    })
    df_reinforce["clean_tweet"] = df_reinforce["tweet"].apply(clean_text)

    df = pd.concat([df, df_neutral, df_reinforce], ignore_index=True)

    # Force all classes to exist
    assert set(df["class"]) == {0, 1, 2}, "Missing class detected"

    # Balance
    max_count = df["class"].value_counts().max()
    df = pd.concat([
        resample(df[df["class"] == c], replace=True, n_samples=max_count, random_state=42)
        for c in [0, 1, 2]
    ])

    return df


data = load_data()

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["clean_tweet"])
y = data["class"].values


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
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


st.title("Toxic Comment Detection")
st.sidebar.write(f"Accuracy: {acc:.2f}")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)


user_text = st.text_area("Enter text")

if st.button("Analyze") and user_text.strip():
    cleaned = clean_text(user_text)
    X_input = vectorizer.transform([cleaned])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]

    labels = {0: "Hate Speech", 1: "Offensive", 2: "Non-toxic"}
    st.write(f"Prediction: **{labels[pred]}** ({prob[pred]:.2f})")
