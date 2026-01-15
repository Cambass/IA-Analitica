# 🗣️ NLP: Word Sense Disambiguation (WSD) Classifier

## 📌 Project Overview
One of the hardest challenges in Natural Language Processing (NLP) is **polysemy**: when a single word has multiple meanings depending on the context.

This project implements a supervised text classifier to solve the Word Sense Disambiguation (WSD) problem for two polysemous target words:
1.  **"Hard"**: Contexts of *difficulty* vs. *physical solidity* vs. *strictness*.
2.  **"Serve"**: Contexts of *service* vs. *legal/time* vs. *sports*.

The solution compares two feature extraction techniques using a **Naive Bayes Classifier**:
* **Bag of Words (BoW):** Frequency-based approach.
* **Context Bigrams:** Capturing local semantic relationships (neighboring words).

*Developed as part of the Artificial Intelligence Specialization at UNIR.*

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **NLP Toolkit:** NLTK (Natural Language Toolkit).
* **Algorithm:** Naive Bayes Classifier (Probabilistic).
* **Concepts:** Tokenization, Feature Engineering, Context Window.

## 🔬 Methodology & Approach

### 1. The Challenge
The algorithm must decide the correct sense tag based on the sentence structure:
> *"The teacher gave a **hard** exam"* → `HARD1` (Difficult)
> *"The rock is **hard**"* → `HARD2` (Physical)

### 2. Feature Engineering
We implemented two strategies to extract features from the training corpus:
* **Simple BoW:** Checks if specific keywords exist in the sentence (e.g., "exam", "rock").
* **Context Window (N-Grams):** Analyzes the words strictly adjacent to the target word. This proved superior for disambiguation as it captures syntax (e.g., "serve **the ball**" vs "serve **dinner**").

### 3. Key Findings
* **Local Context Matters:** Approaches that consider the immediate neighbors (Bigrams) outperformed global BoW for specific verbs like "serve".
* **Data Scarcity:** The Naive Bayes model showed high sensitivity to unseen vocabulary, highlighting the need for larger corpora or embeddings (Word2Vec/BERT) in production environments.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/camilo-ferro/NLP-Word-Sense-Disambiguation-WSD.git](https://github.com/camilo-ferro/NLP-Word-Sense-Disambiguation-WSD.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    ```bash
    jupyter notebook notebooks/wsd_naive_bayes_lab.ipynb
    ```

## 📂 Project Structure
* `notebooks/`: Contains the implementation and training logic.
* `docs/`: Detailed analysis of the ambiguous terms and error analysis.

---
**Author:** Camilo Ferro
*Specialization in Artificial Intelligence - UNIR*
