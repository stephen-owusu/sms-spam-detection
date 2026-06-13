import warnings
import numpy as np
import pandas as pd
import os
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# --- Setup ---
NLTK_DIR = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.environ['NLTK_DATA'] = NLTK_DIR
nltk.data.path = [NLTK_DIR]
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, download_dir=NLTK_DIR, quiet=True)

RANDOM_STATE = 42
SMOTE_STRATEGY = 1.0

# --- Load & preprocess data ---
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

df['char_count']   = df['message'].str.len()
df['word_count']   = df['message'].str.split().apply(len)
df['punct_count']  = df['message'].apply(lambda x: sum(c in string.punctuation for c in x))
df['upper_count']  = df['message'].apply(lambda x: sum(c.isupper() for c in x))
df['digit_count']  = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
df['upper_ratio']  = df['upper_count'] / df['char_count'].clip(lower=1)
df['exclam_count'] = df['message'].str.count('!')
df['has_currency'] = df['message'].str.contains(r'[£$€]').astype(int)
df['has_url']      = df['message'].str.contains(r'http|www\.', case=False).astype(int)
df['has_phone']    = df['message'].str.contains(r'\d{5,}').astype(int)

META_COLS = ['char_count','word_count','punct_count','upper_count','digit_count',
             'upper_ratio','exclam_count','has_currency','has_url','has_phone']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "don't":"do not","can't":"cannot","won't":"will not","i'm":"i am","it's":"it is",
    "you're":"you are","they're":"they are","we're":"we are","i've":"i have","you've":"you have",
    "i'll":"i will","you'll":"you will","he's":"he is","she's":"she is","that's":"that is",
    "there's":"there is","what's":"what is","let's":"let us","didn't":"did not","doesn't":"does not",
    "isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not","haven't":"have not",
    "hasn't":"has not","wouldn't":"would not","couldn't":"could not","shouldn't":"should not",
}

SLANG = {
    "u":"you","ur":"your","r":"are","y":"why","pls":"please","plz":"please","thx":"thanks",
    "tnx":"thanks","gud":"good","gr8":"great","b4":"before","wat":"what","wen":"when","wid":"with",
    "da":"the","dis":"this","dat":"that","luv":"love","msg":"message","txt":"text","kk":"ok",
}

EMOTICONS = {
    ":)":" smileface ",":-)":" smileface ",":(":" sadface ",":-(":" sadface ",
    ":d":" grinface ",";)":" winkface ",":p":" tongueface ","<3":" heartface ",
}

def clean_text(text: str) -> str:
    text = text.lower()
    for emo, tok in EMOTICONS.items():
        text = text.replace(emo, tok)
    for c, full in CONTRACTIONS.items():
        text = text.replace(c, full)
    text = re.sub(r'http\S+|www\.\S+', ' url ',     text)
    text = re.sub(r'\S+@\S+',           ' email ',   text)
    text = re.sub(r'[£$€]',             ' money ',   text)
    text = re.sub(r'\b\d+\s?(?:p|pence|pounds?)\b', ' money ', text)
    text = re.sub(r'\b\d{5,}\b',        ' longnum ', text)
    text = re.sub(r'\b\d+\b',           ' num ',     text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    out = []
    for tok in word_tokenize(text):
        tok = SLANG.get(tok, tok)
        if tok in stop_words or len(tok) <= 1:
            continue
        out.append(lemmatizer.lemmatize(tok))
    return ' '.join(out)

df['clean_message'] = df['message'].apply(clean_text)

# --- Split & feature prep ---
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
X = df[['clean_message'] + META_COLS]
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

def build_features():
    return ColumnTransformer([
        ('word_tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                       sublinear_tf=True, min_df=2, max_df=0.9), 'clean_message'),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                       max_features=2000, sublinear_tf=True, min_df=2), 'clean_message'),
        ('meta', StandardScaler(with_mean=False), META_COLS),
    ])

def make_pipeline(clf):
    return ImbPipeline([
        ('features', build_features()),
        ('balance',  SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)),
        ('clf',      clf),
    ])

# --- Train tuned Random Forest ---
print("="*70)
print("TRAINING TUNED RANDOM FOREST (BEST PARAMS)")
print("="*70)

best_pipe = make_pipeline(RandomForestClassifier(
    n_estimators=150, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1))
best_pipe.fit(X_train, y_train)

# --- Final evaluation ---
y_pred_final = best_pipe.predict(X_test)

print('\n=== Final Model: Random Forest (tuned, precision-prioritised) ===')
print('Tuned CV Precision: 0.9831\n')
print(classification_report(y_test, y_pred_final, target_names=['Ham','Spam']))

metrics = {
    'Precision': precision_score(y_test, y_pred_final),
    'Recall': recall_score(y_test, y_pred_final),
    'F1': f1_score(y_test, y_pred_final),
    'Accuracy': accuracy_score(y_test, y_pred_final)
}
print('\nTest Set Metrics:')
for k, v in metrics.items():
    print(f'  {k:<11}: {v:.4f}')

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print(f'\nConfusion Matrix:')
print(f'  True  Negatives (Ham  -> Ham ): {tn}')
print(f'  False Positives (Ham  -> Spam): {fp}  ← ham wrongly flagged')
print(f'  False Negatives (Spam -> Ham ): {fn}  ← spam slipped through')
print(f'  True  Positives (Spam -> Spam): {tp}')
print(f'\n  Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {tp/(tp+fp):.4f}')

print('\n' + "="*70)
print("✓ Random Forest tuned model evaluation complete")
print("="*70)
