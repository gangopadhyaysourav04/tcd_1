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

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Toxic Comment Detection", page_icon="☣️")


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()

    mappings = {
        "g@ndu": "gandu", "g@ndoo": "gandu", "g4ndu": "gandu",
        "chutiyaa": "chutiya", "chut!ya": "chutiya", "chutiy@": "chutiya",
        "madarch0d": "madarchod", "bhench0d": "bhenchod",
        "maderchod": "madarchod",
        "fck": "fuck", "fucc": "fuck", "fak": "fuck",
        "laude": "lauda", "lawde": "lauda", "lodu": "lauda",
        "lund": "lauda", "chut": "chut", "chutmarike": "chutmarike",
        "kutter ka baccha": "kutta", "kutte ka baccha": "kutta",
        "bhen ke lode": "bhenchod", "bkl": "bewakoof"
    }

    for k, v in mappings.items():
        text = text.replace(k, v)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Load + build dataset (CACHED)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_data_with_more_hinglish_final_v2_cleaned.csv")
    df = df[["class", "tweet"]].dropna()
    df["clean_tweet"] = df["tweet"].apply(clean_text)

    # Neutral injections (FIXED commas)
    neutral_comments = [
        "good morning", "good night", "have a nice day", "thank you",
        "thanks bro", "thanks bhai", "love you bro", "hope you are fine",
        "how are you", "take care", "good evening",
        "you are amazing", "nice work", "great job", "congratulations",
        "happy birthday", "best wishes", "keep going", "awesome",
        "fantastic", "you are welcome", "good luck", "stay safe",
        "peace", "jai hind", "namaste", "radhe radhe", "shukriya",
        "dhanyawaad", "jai shree ram", "hello bhai", "good vibes only",
        "allah hu akbar", "salaam alaikum", "ram ram", "om shanti",
        "sat sri akaal", "namaskar", "jai mata di", "jai bajrangbali",
        "hare krishna", "kaise ho", "kya haal hai", "sab badiya",
        "theek ho na", "sab theek hai",
        "asha karta hoon ki aapka din khushalpurwak beetein",
        "namaste bhai", "radhe radhe bhai", "ram ram bhai",
        "jai shree ram bhai", "kaise ho bhai", "badiya ho", "mast ho",
        "all good bro", "kya scene hai", "milte hai jaldi",
        "chalo milte hai", "kya haal chaal", "sabka bhala ho"
    ]

    df_neutral = pd.DataFrame({
        "class": [2] * len(neutral_comments),
        "tweet": neutral_comments
    })
    df_neutral["clean_tweet"] = df_neutral["tweet"].apply(clean_text)

    # Reinforcement abuse
    reinforcement = [
        "madarchod hai tu", "lawde ke bacche", "bhenchod sala",
        "gandu aadmi", "chutiya insaan", "bc mc",
        "madarchod lawda", "teri maa ka bhosda",
        "laude", "maderchod", "kutter ka baccha",
        "lund", "chut", "chutmarike"
    ]

    df_reinforce = pd.DataFrame({
        "class": [1] * len(reinforcement),
        "tweet": reinforcement
    })
    df_reinforce["clean_tweet"] = df_reinforce["tweet"].apply(clean_text)

    df = pd.concat([df, df_neutral, df_reinforce], ignore_index=True)

    # HARD validation
    classes, counts = np.unique(df["class"], return_counts=True)
    if len(classes) < 3:
        raise ValueError(f"Dataset corrupted. Classes found: {classes}")

    # Balance classes
    max_count = counts.max()
    balanced_df = pd.concat([
        resample(
            df[df["class"] == c],
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        for c in classes
    ])

    balanced_df["class_name"] = balanced_df["class"].map({
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Non-toxic"
    })

    return balanced_df


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
def feature_engineering(data):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words="english",
        ngram_range=(1, 3)
    )

    X = vectorizer.fit_transform(data["clean_tweet"])
    y = data["class"].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Target has only one class")

    return X, y, vectorizer


# --------------------------------------------------
# Model training (NO CACHE – deliberate)
# --------------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train has only one class – cannot train")

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return (
        model,
        accuracy_score(y_test, preds),
        classification_report(
            y_test, preds,
            target_names=["Hate Speech", "Offensive", "Non-toxic"],
            output_dict=True
        ),
        confusion_matrix(y_test, preds)
    )


# --------------------------------------------------
# Pipeline execution
# --------------------------------------------------
data = load_data()
X, y, tfidf_vectorizer = feature_engineering(data)
model, acc, report, cm = train_model(X, y)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Toxic Comment Detection (Hinglish + English)")
st.markdown("Robust multi-class toxic language classifier")

st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: **{acc:.2f}**")

st.sidebar.subheader("F1 Scores")
st.sidebar.write(f"Hate Speech: {report['Hate Speech']['f1-score']:.2f}")
st.sidebar.write(f"Offensive: {report['Offensive']['f1-score']:.2f}")
st.sidebar.write(f"Non-toxic: {report['Non-toxic']['f1-score']:.2f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Hate", "Offensive", "Non-toxic"],
    yticklabels=["Hate", "Offensive", "Non-toxic"],
    ax=ax
)
st.pyplot(fig)

st.subheader("Class Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(x="class_name", data=data, ax=ax2)
st.pyplot(fig2)

st.header("Try a Custom Comment")

user_text = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if user_text.strip():
        cleaned = clean_text(user_text)
        X_input = tfidf_vectorizer.transform([cleaned])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        class_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Non-toxic"}
        label = class_map[pred]
        conf = prob[pred]

        if pred == 0:
            st.error(f"{label} ({conf:.2f})")
        elif pred == 1:
            st.warning(f"{label} ({conf:.2f})")
        else:
            st.success(f"{label} ({conf:.2f})")
    else:
        st.info("Enter some text.")

    else:
        st.info("Please enter text above to analyze.")

