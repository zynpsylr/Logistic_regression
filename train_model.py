import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import re

# Veriyi oku
data = pd.read_csv('IMDB Dataset.csv')

# 'positive' etiketini 1, 'negative' etiketini 0 olacak şekilde dönüştür
data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Yorumları temizle
data['clean_review'] = data['review'].apply(clean_text)

# Değişkenleri ayır
X = data['clean_review']
Y = data['label']

# Eğitim ve test seti
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# TF-IDF ile metni sayısallaştır
vectorizer=TfidfVectorizer(max_features=5000)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Lojistik regresyon modelini eğit
model = LogisticRegression()
model.fit(x_train_vec,y_train)

# Lojistik regresyon modelini eğit
y_pred = model.predict(x_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Modeli ve vektörleştiriciyi kaydet
joblib.dump(model,'logistic_regression.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')