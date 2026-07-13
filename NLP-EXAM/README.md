# Spam Detection NLP Project 🚀

A complete natural language processing project for detecting spam messages using machine learning and deep learning models, with a bonus Flask web application for real-time testing.

## Overview

This project implements a **Telegram Spam Detection System** using a dataset from Hugging Face. It includes multiple models (Naive Bayes & LSTM) with a complete NLP pipeline from data preprocessing to web deployment.

## Features

✅ **Complete NLP Pipeline**
- Text preprocessing (lowercase, punctuation removal, lemmatization)
- POS tagging and linguistic analysis
- TF-IDF vectorization

✅ **Machine Learning Models**
- Naive Bayes classifier (fast & efficient)
- LSTM deep learning model (context-aware)
- Model evaluation with confusion matrix & ROC curves

✅ **Web Application**
- Flask-based REST API
- Beautiful HTML interface
- Real-time spam detection demo

## Project Structure

```
.
├── spam_detection_project.ipynb    # Main Jupyter notebook with full analysis
├── train_model.py                  # Script to train and save models
├── app.py                          # Flask web application
├── templates/
│   └── index.html                  # Web UI
├── dataset.csv                     # Training data
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### Option A: Run Jupyter Notebook (Main Analysis)

1. Open Google Colab or Jupyter Notebook
2. Upload `spam_detection_project.ipynb`
3. Run all cells to see visualizations (confusion matrix, ROC curves)

### Option B: Run Flask Web Application

#### Prerequisites
- Python 3.7+
- pip

#### Setup & Run

```bash
# 1. Navigate to project directory
cd NLP-EXAM

# 2. Create virtual environment (optional but recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (creates spam_model.pkl & vectorizer.pkl)
python train_model.py

# 6. Run the Flask app
python app.py

# 7. Open your browser to: http://127.0.0.1:5000
```

## Testing the Application

Once the Flask app is running, try these examples:

- **SPAM Example:** "You won a free prize call now" → Result: 🔴 SPAM
- **HAM Example:** "Hello how are you today" → Result: 🟢 HAM

## Technical Details

### NLP Pipeline Components

1. **Preprocessing:**
   - Remove punctuation and special characters
   - Convert to lowercase
   - Lemmatization to get word stems

2. **Feature Engineering:**
   - TF-IDF vectorization
   - POS tag analysis
   - Frequent adjective extraction (e.g., "free", "best" in spam)

3. **Models:**
   - **Naive Bayes:** Primary model - fast and effective for text classification
   - **LSTM:** Deep learning model - captures sequential context

### Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib & seaborn (visualization)
- nltk (natural language processing)
- tensorflow (deep learning)
- gensim (word embeddings)
- flask (web framework)

## Results

The Naive Bayes model achieves high accuracy on the spam detection task with real-time inference capabilities.

## Files Description

| File | Purpose |
|------|---------|
| `spam_detection_project.ipynb` | Complete analysis, data exploration, and model training |
| `train_model.py` | Standalone script to train and pickle models |
| `app.py` | Flask web server and API endpoints |
| `templates/index.html` | Frontend interface for the web app |
| `requirements.txt` | Python package dependencies |
| `dataset.csv` | Training and testing data |

## Troubleshooting

**Error: "Model files not found"**
- Run `python train_model.py` to generate `spam_model.pkl` and `vectorizer.pkl`

**Port 5000 already in use?**
- Modify the port in `app.py`: `app.run(debug=True, port=5001)`

**Installation issues?**
- Use Google Colab instead for instant setup (just upload the notebook)

## Author

**Name:** [Your Name]  
**Group:** [Your Group]  
**Project Type:** University NLP Mini Project

## License

MIT License - feel free to use and modify this project.

---

**Good luck! 🎉**
