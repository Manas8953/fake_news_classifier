# 📰 Fake News Classification Using ML & Deep Learning

This project focuses on building a robust text classification system to detect **fake news articles**. Utilizing both traditional machine learning and deep learning techniques, we compare multiple models to find the most effective approach. The **LSTM model** emerged as the top performer, achieving an impressive **99.18% accuracy**.

---

## 📌 Project Overview

* **Dataset Size:** 20,800 rows
* **Source:** Kaggle Fake News Dataset
* **Features:** `id`, `title`, `author`, `text`
* **Target:** `label` (1 for fake, 0 for real)

---

## 🧠 Problem Statement

Fake news has become a significant challenge in the digital age, contributing to misinformation and social unrest. This project aims to automatically classify news articles as *real* or *fake* using text-based features and natural language processing (NLP).

---

## ✅ Objectives

* Clean and preprocess raw textual data.
* Apply **feature engineering** and **text vectorization** techniques.
* Compare multiple machine learning and deep learning models.
* Evaluate and visualize model performance.
* Deploy the most accurate model for real-world prediction.

---

## 🛠️ Technologies Used

* **Languages:** Python
* **Libraries:** Pandas, Numpy, Scikit-learn, NLTK, Matplotlib, Seaborn, XGBoost, CatBoost, TensorFlow/Keras
* **Models Compared:**

  * Logistic Regression
  * Naive Bayes
  * Decision Trees
  * Random Forest
  * XGBoost
  * CatBoost
  * LSTM (Long Short-Term Memory)

---

## ⚙️ Methodology

### 1. Data Collection & Cleaning

* Imported 20,800 rows with 4 features and 1 target column.
* Removed null values, irrelevant columns, and duplicate entries.

### 2. Preprocessing & Feature Engineering

* Lowercasing, removing punctuation, stopwords, and lemmatization.
* Applied **TF-IDF vectorization** for traditional ML models.
* Used **Tokenization + Padding** for LSTM input.

### 3. Model Training & Evaluation

* Trained and tested all ML models using cross-validation and accuracy/F1 metrics.
* LSTM model built using Keras with embedding and LSTM layers.
* Achieved:

  * ✅ **99.18% Accuracy with LSTM**
  * ✅ LSTM significantly outperformed other models in terms of both accuracy and generalization.

---

## 📈 Results

| Model               | Accuracy     |
| ------------------- | ------------ |
| Logistic Regression | \~92%        |
| Naive Bayes         | \~93%        |
| Random Forest       | \~94%        |
| XGBoost             | \~96%        |
| CatBoost            | \~96.5%      |
| **LSTM**            | **99.18%** ✅ |
