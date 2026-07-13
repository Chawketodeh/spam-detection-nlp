from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize App
app = Flask(__name__)

# Load Model & Vectorizer
try:
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Please run 'train_model.py' first.")
    exit()

# Preprocessing Setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        data = [processed_message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        
        result = "Spam" if prediction[0] == 1 else "Ham (Not Spam)"
        color = "red" if prediction[0] == 1 else "green"
        
        return render_template('index.html', prediction=result, message=message, color=color)

if __name__ == '__main__':
    app.run(debug=True)
