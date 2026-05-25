# 🎬 IMDB Sentiment Analysis using Recurrent Neural Network

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://rnnimbdreview-byhima.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Is this movie review Positive or Negative?** — An end-to-end RNN model trained on 50,000 real IMDB reviews that understands the sequence and context of language.

---

## 🚀 Live Demo

🔗 **[Try it Live → rnnimbdreview-byhima.streamlit.app](https://rnnimbdreview-byhima.streamlit.app/)**

---

## 📌 Problem Statement

ANN treats every word independently — it misses context. Language is sequential. The word **"not"** completely changes the meaning of a sentence:

> *"The movie was good"* → Positive  
> *"The movie was not good"* → Negative

RNNs solve this by maintaining a **hidden state** — memory that passes from word to word, capturing sequence and context.

---

## 🏗️ Project Architecture

```
Raw Text Input
      ↓
Tokenization + Padding
      ↓
Embedding Layer (10,000 vocab → 128 dims)
      ↓
SimpleRNN(128) ← Hidden state passes word to word
      ↓
Dense(1, Sigmoid) ← Output
      ↓
Positive (1) / Negative (0)
```

---

## 📊 Dataset

- **Source:** IMDB Dataset (built into Keras)
- **Size:** 50,000 movie reviews (25K train + 25K test)
- **Labels:** Positive (1) / Negative (0)
- **Vocab Size:** Top 10,000 most frequent words
- **Max Sequence Length:** 500 words

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Dataset | Keras IMDB built-in |
| Preprocessing | Tokenization, Padding |
| Deployment | Streamlit |

---

## 🔬 What Happens Under the Hood

### 1. Data Loading + Preprocessing
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to same length
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)
```

### 2. RNN Architecture
```python
model = Sequential([
    Embedding(10000, 128, input_length=500),
    SimpleRNN(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 3. How RNN Processes Text
```
"The movie was not good"
         ↓
[Token1] → hidden_state_1
[Token2] → hidden_state_2 (remembers Token1)
[Token3] → hidden_state_3 (remembers Token1+2)
[Token4] → hidden_state_4 (remembers all)
[Token5] → Final hidden state → Prediction
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~85% |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Vocab Size | 10,000 |

---

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/rnn-imdb-sentiment
cd rnn-imdb-sentiment

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
rnn-imdb-sentiment/
│
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── model.h5            # Saved RNN model
└── requirements.txt    # Dependencies
```

---

## 🧠 Key Learnings

- Why ANN fails for sequential data
- How RNN hidden state passes information forward
- Embedding layers — converting words to dense vectors
- Vanishing gradient problem in simple RNNs
- Why LSTM/GRU were invented (next project!)

---

## 👨‍💻 Author

**Himanshu Bendale**
- 🎓 B.E. AI & DS — Mumbai University
- 🎓 B.S. Electronic Systems — IIT Madras
- 🔗 [GitHub](https://github.com/yourusername)
- 💼 [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🗺️ Roadmap

```
✅ ANN — Churn Prediction (deployed)
✅ RNN — Sentiment Analysis (deployed)
✅ LSTM — Next Word Predictor (deployed)
🔜 Transformers — Attention Mechanism
🔜 LLM Fine-Tuning — LoRA/QLoRA on Llama 3
```

> *"The grind is on."* 💪
