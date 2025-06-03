import streamlit as st
import pandas as pd 
import joblib
import re

# Eğitilmiş model ve vektörleştiriciyi yükle
model = joblib.load("logistic_regression.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🎬 IMDb Review Sentiment Analysis")

# Kullanıcıdan yorum al
user_input = st.text_area("Enter your movie review in English:")
st.caption("⚠️ Lütfen yorumunuzu İngilizce olarak giriniz. Model sadece İngilizce yorumlarla eğitildi.")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
    text = re.sub(r'[^a-z\s]', '', text)  # Sadece harf ve boşluk bırak
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları kaldır
    return text

cleaned_input = clean_text(user_input)

if st.button('Predict'):
    vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(vec)[0]

# Sonucu kullanıcıya göster
    if prediction == 1:
        st.write("🌟 Positive Review")
    else:
        st.write("💢 Negative Review")
