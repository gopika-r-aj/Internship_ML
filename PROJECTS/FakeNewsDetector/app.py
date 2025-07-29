import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv
from datetime import datetime
import pandas as pd

# ------------------ Setup ------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ------------------ CSS Styling ------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2);
            font-family: 'Segoe UI', sans-serif;
            color: #111 !important;
        }
        .main-title {
            text-align: center;
            font-size: 50px;
            font-weight: 800;
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #333;
            margin-bottom: 40px;
        }
        textarea {
            background-color: #ffffff !important;
            color: #111 !important;
            border-radius: 8px !important;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
        }
        label, .stTextInput>div>div>input, .stTextArea textarea {
            color: #222 !important;
        }
        .css-1n76uvr p {
            color: #222 !important;
            font-weight: 600;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 2px 15px rgba(0,0,0,0.2);
            background-color: white;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin-top: 30px;
            color: #111;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #555;
            margin-top: 70px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            padding: 0.6em 1.2em;
            border-radius: 6px;
            border: none;
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        
        .css-16huue1 span {
            color: #222 !important;
            font-weight: 600;
        }

        .css-1r6slb0 {
            color: #ffffff !important;
            font-weight: 600;
        }

    </style>
""", unsafe_allow_html=True)

# ------------------ Clean Function ------------------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# ------------------ UI ------------------
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Check whether a news article is Real ✅ or Fake ❌ using Machine Learning</div>', unsafe_allow_html=True)

user_input = st.text_area("🔍 Enter news headline or article:")

# ------------------ Prediction ------------------
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article to analyze.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input]).toarray()
        prediction = model.predict(vectorized_input)[0]
        confidence = model.predict_proba(vectorized_input)[0]

        # Store results in session state to persist
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.user_input = user_input
        st.session_state.cleaned_input = cleaned_input

        # Save to log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = "FAKE" if prediction == 1 else "REAL"
        with open("prediction_log.csv", mode="a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, user_input[:70], label, round(confidence[0]*100, 2), round(confidence[1]*100, 2)])

# ------------------ Show Result if Prediction Exists ------------------
if 'prediction' in st.session_state:
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence
    cleaned_input = st.session_state.cleaned_input
    user_input = st.session_state.user_input

    if prediction == 1:
        st.markdown('<div class="prediction-box" style="border-left: 8px solid red;">❌ This news is likely <span style="color:red;">FAKE</span>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box" style="border-left: 8px solid green;">✅ This news is likely <span style="color:green;">REAL</span>.</div>', unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center;'>Confidence Score → Real: <b>{confidence[0]*100:.2f}%</b> | Fake: <b>{confidence[1]*100:.2f}%</b></div>", unsafe_allow_html=True)

    # Highlight keywords
    input_words = cleaned_input.split()
    feature_names = vectorizer.get_feature_names_out()
    input_tfidf = vectorizer.transform([cleaned_input])
    scores = dict(zip(feature_names, input_tfidf.toarray()[0]))
    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    highlighted_text = ""
    for word in input_words:
        if word in dict(sorted_keywords[:5]):
            highlighted_text += f"<span style='background-color: #ffe066; padding: 2px 5px; border-radius: 3px;'>{word}</span> "
        else:
            highlighted_text += word + " "
    st.markdown("### 🔍 Key Influential Words in Prediction")
    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 8px;'>{highlighted_text}</div>", unsafe_allow_html=True)

# ------------------ Prediction History Viewer ------------------
st.markdown("### 📂 View Prediction History")

# Custom label + checkbox in two columns (dark label)
col1, col2 = st.columns([0.1, 0.9])
with col1:
    show_log = st.checkbox("", key="show_history_box")
with col2:
    st.markdown('<span style="font-size:17px; font-weight:600; color:#222;">Show Prediction History</span>', unsafe_allow_html=True)

if show_log:
    try:
        log_df = pd.read_csv("prediction_log.csv", names=["Time", "Input Snippet", "Prediction", "Real %", "Fake %"])
        if len(log_df) > 0:
            st.dataframe(log_df.tail(10))

            # Use inline HTML + styled container for download button
            st.markdown("""
                <style>
                div[data-testid="stDownloadButton"] > button {
                    background-color: #3498db;
                    color: white !important;
                    font-weight: 600;
                    border-radius: 6px;
                    padding: 0.5em 1.2em;
                }
                div[data-testid="stDownloadButton"] > button:hover {
                    background-color: #2980b9;
                    color: white !important;
                }
                </style>
            """, unsafe_allow_html=True)

            st.download_button(
                label="⬇️ Download Full Log",
                data=log_df.to_csv(index=False),
                file_name="prediction_log.csv",
                mime="text/csv"
            )
        else:
            st.info("Prediction log is currently empty.")
    except FileNotFoundError:
        st.info("No predictions made yet. Make some to view history.")


# ------------------ Footer ------------------
st.markdown('<div class="footer">Made by Gopika Raj | AIML Internship Project</div>', unsafe_allow_html=True)


