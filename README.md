# SMS Spam Classifier

A machine learning model that classifies SMS messages as spam or ham (legitimate), with multilingual support for English, Hindi, and Hinglish.

## Overview

This project started with a baseline TF-IDF + Naive Bayes approach and was later upgraded to multilingual sentence embeddings for better real-world performance — especially on Indian SMS data, where messages often mix Hindi and English.

## Features
- **Multilingual embeddings**: Uses `paraphrase-multilingual-MiniLM-L12-v2` to generate sentence-level representations, enabling classification across English, Hindi, and Hinglish text.
- **Class imbalance handling**: Logistic Regression with `class_weight='balanced'` to handle the natural skew between spam and ham messages.
- **Clean preprocessing pipeline**: Tuned to avoid common bugs like incorrect token merging during text cleaning.

## Tech Stack
- Python
- scikit-learn (Logistic Regression)
- Sentence-Transformers (multilingual MiniLM embeddings)
- Pandas / NumPy

## Model Evolution
| Version | Approach | Notes |
|---------|----------|-------|
| v1 | TF-IDF + Multinomial Naive Bayes | Baseline, English-only |
| v2 | Multilingual Sentence Embeddings + Logistic Regression | English, balanced classes |

## How It Works
1. Input SMS text is cleaned and preprocessed
2. Text is converted into sentence embeddings using the multilingual MiniLM model
3. Embeddings are passed to a trained Logistic Regression model
4. Model outputs a prediction: **Spam** or **Ham**
```

## Future Work
- Deploy as a web app (Flask/Streamlit)
- Expand dataset with more Hinglish examples
- Add SHAP-based explainability for predictions

## License
This project is open source and available under the [MIT License](LICENSE).
