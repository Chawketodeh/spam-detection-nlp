import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already present)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# 1. Load Data
url = "https://huggingface.co/datasets/thehamkercat/telegram-spam-ham/resolve/main/dataset.csv"
print("Loading dataset...")
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess_text)

# 3. Train Model
X = df['clean_text']
y = df['text_type'].map({'ham': 0, 'spam': 1})

cv = CountVectorizer()
X_cv = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
print("Training model...")
clf.fit(X_train, y_train)
print(f"Model Accuracy: {clf.score(X_test, y_test)}")

# 4. Save Model & Vectorizer
print("Saving model and vectorizer...")
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("Done! Files 'spam_model.pkl' and 'vectorizer.pkl' created.")
