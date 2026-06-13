# SMS Spam Detection (NLP)

Binary text classification project that detects spam SMS messages, built on the
SMS Spam Collection dataset (~5,500 labelled messages). The pipeline prioritises
**precision** so legitimate messages are almost never wrongly blocked.

## Contents

| File | Description |
|------|-------------|
| `spam_project_precision.ipynb` | Main end-to-end notebook: EDA, preprocessing, feature engineering, SMOTE balancing, model comparison, GridSearchCV tuning, final evaluation |
| `spam_project.ipynb` | Earlier version of the notebook |
| `demo_app.py` | **SpamGuard** — Streamlit app for single-message and bulk screening |
| `spam.csv` | SMS Spam Collection dataset |
| `run_demo.bat` | One-click launcher for the demo app |

## Approach

- Text preprocessing: placeholder tokens (URL / money / numbers), contraction and
  slang expansion, emoticon handling, lemmatization
- Features: word TF-IDF + character n-grams + 10 hand-crafted meta-features
- Class balancing: SMOTE (training split only)
- Models compared: Naive Bayes, Logistic Regression, Linear SVM, Random Forest,
  Gradient Boosting — tuned with GridSearchCV scored on precision

## Run the demo app

```
pip install streamlit scikit-learn imbalanced-learn nltk pandas
streamlit run demo_app.py
```

The app builds its classifier from `spam.csv` on first launch (cached afterwards)
and supports single-message checks plus bulk screening via CSV/TXT upload.
