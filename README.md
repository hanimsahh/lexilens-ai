# 🧠 LexiLens AI

AI-powered handwriting pattern analysis platform supporting dyslexia screening through machine learning and AI-assisted interpretation.

---

## 📌 Overview

LexiLens AI is a final-year university project designed to explore how machine learning and AI can assist with identifying handwriting reversal patterns commonly associated with dyslexia.

The platform allows users to:

* Upload handwriting samples for AI analysis
* Detect reversal and correction patterns
* View confidence scores and visual insights
* Complete guided screening tests
* Interact with an AI educational assistant
* Generate downloadable PDF reports

> ⚠️ LexiLens AI is an educational screening support tool only and is **not** a clinical diagnostic system.

---

## ✨ Features

### 🔍 Handwriting Analysis

* Upload handwritten letters or numbers
* AI classification into:

  * Normal
  * Reversal
  * Corrected
* Confidence score visualisation
* Age-contextual interpretation

### 🎯 Guided Screening Test

* Step-by-step structured handwriting screening
* Tracks reversal frequency
* Generates risk-level summaries
* Educational recommendations

### 🤖 AI Assistant

* Dyslexia-focused educational chatbot
* Explains handwriting reversal patterns
* Interprets screening results responsibly
* Provides supportive guidance

### 📊 Model Insights Dashboard

* Confusion matrices
* ROC / AUC analysis
* Calibration curves
* Learning curves
* CNN training metrics
* PCA / t-SNE visualisations

### 📄 Reporting System

* PDF report generation
* Email delivery support
* Prediction history tracking
* SQLite-based storage

---

## 🧠 Machine Learning Models

Multiple machine learning models were evaluated during development:

| Model                  | Accuracy |
| ---------------------- | -------- |
| Random Forest          | 85.85%   |
| SVM (Cross Validation) | 90.01%   |
| CNN                    | 76.45%   |
| Gradient Boosting      | 75.90%   |
| Logistic Regression    | 70.51%   |

The deployed system uses a calibrated Random Forest classifier.

---

## 🛠️ Tech Stack

### Frontend

* Streamlit
* Custom CSS UI
* Matplotlib

### Backend

* Python
* SQLite
* Scikit-learn
* Joblib

### AI & APIs

* Anthropic Claude API
* Resend Email API

---

## 📂 Project Structure

```bash
lexilens-ai/
│
├── app.py
├── requirements.txt
├── dyslexia_finall.ipynb
│
├── assets/
│   ├── graphs/
│   └── examples/
│
├── .gitignore
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/hanimsahh/lexilens-ai.git
cd lexilens-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_key_here
RESEND_API_KEY=your_key_here
```

---

## 📈 Dataset

The project was trained using a handwriting image dataset containing over 150,000 images focused on dyslexia-related reversal patterns.

The dataset is not included in this repository due to size limitations.

---

## ⚠️ Ethical Considerations

This project was developed with strong emphasis on ethical AI principles.

The system:

* Does not provide medical diagnoses
* Avoids deterministic language
* Acknowledges prediction uncertainty
* Encourages professional educational assessment where appropriate
* Uses supportive and non-alarmist communication

---

## 🎓 Academic Context

This project was developed as part of a Final Year Project focused on:

* Educational technology
* Machine learning
* Human-computer interaction
* AI ethics
* Dyslexia screening support systems

---

### Dashboard

<img width="1200" alt="LexiLens Dashboard" src="https://github.com/user-attachments/assets/example-dashboard" />

### Model Insights

<img width="1200" alt="Model Insights" src="https://github.com/user-attachments/assets/example-insights" />

---

## 📄 License

This project is for educational and research purposes only.

---

## 👤 Author

Developed by Hanimsah Tezel

Final Year Project · University of Huddersfield
