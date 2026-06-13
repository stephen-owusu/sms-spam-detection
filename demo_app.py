"""
SpamGuard — Message Screening
=============================
Production-style Streamlit app that screens SMS / text messages and flags
likely spam before it reaches the inbox. Supports single-message checks and
bulk screening via file upload, with downloadable results.

The classifier is built on first launch from spam.csv (cached afterwards),
using the tuned pipeline selected in spam_project_precision.ipynb.

Run with:
    py -3.14 -m streamlit run demo_app.py
"""
import io
import os
import re
import string
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / 'spam.csv'
HISTORY_LIMIT = 25

# --- NLTK data (same non-redirected folder the notebook uses) ----------------
NLTK_DIR = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(NLTK_DIR, exist_ok=True)
os.environ['NLTK_DATA'] = NLTK_DIR
nltk.data.path = [NLTK_DIR]
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, download_dir=NLTK_DIR, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Preprocessing (identical to the notebook) --------------------------------
CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not", "i'm": "i am", "it's": "it is",
    "you're": "you are", "they're": "they are", "we're": "we are", "i've": "i have", "you've": "you have",
    "i'll": "i will", "you'll": "you will", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "what's": "what is", "let's": "let us", "didn't": "did not", "doesn't": "does not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
}
SLANG = {
    "u": "you", "ur": "your", "r": "are", "y": "why", "pls": "please", "plz": "please", "thx": "thanks",
    "tnx": "thanks", "gud": "good", "gr8": "great", "b4": "before", "wat": "what", "wen": "when", "wid": "with",
    "da": "the", "dis": "this", "dat": "that", "luv": "love", "msg": "message", "txt": "text", "kk": "ok",
}
EMOTICONS = {
    ":)": " smileface ", ":-)": " smileface ", ":(": " sadface ", ":-(": " sadface ",
    ":d": " grinface ", ";)": " winkface ", ":p": " tongueface ", "<3": " heartface ",
}

META_COLS = ['char_count', 'word_count', 'punct_count', 'upper_count', 'digit_count',
             'upper_ratio', 'exclam_count', 'has_currency', 'has_url', 'has_phone']


def clean_text(text: str) -> str:
    text = text.lower()
    for emo, tok in EMOTICONS.items():
        text = text.replace(emo, tok)
    for c, full in CONTRACTIONS.items():
        text = text.replace(c, full)
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'[£$€]', ' money ', text)
    text = re.sub(r'\b\d+\s?(?:p|pence|pounds?)\b', ' money ', text)
    text = re.sub(r'\b\d{5,}\b', ' longnum ', text)
    text = re.sub(r'\b\d+\b', ' num ', text)
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


def featurize(messages: list[str]) -> pd.DataFrame:
    """Raw messages -> the exact feature columns the pipeline expects."""
    return pd.DataFrame([{
        'clean_message': clean_text(m),
        'char_count':    len(m),
        'word_count':    len(m.split()),
        'punct_count':   sum(c in string.punctuation for c in m),
        'upper_count':   sum(c.isupper() for c in m),
        'digit_count':   sum(c.isdigit() for c in m),
        'upper_ratio':   sum(c.isupper() for c in m) / max(len(m), 1),
        'exclam_count':  m.count('!'),
        'has_currency':  int(bool(re.search(r'[£$€]', m))),
        'has_url':       int(bool(re.search(r'http|www\.', m, re.I))),
        'has_phone':     int(bool(re.search(r'\d{5,}', m))),
    } for m in messages])


# --- Classifier (tuned pipeline selected in the notebook) ----------------------
@st.cache_resource(show_spinner='Preparing the screening service (first launch only)...')
def load_classifier():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = featurize(df['message'].tolist())
    y = (df['label'] == 'spam').astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pipe = ImbPipeline([
        ('features', ColumnTransformer([
            ('word_tfidf', TfidfVectorizer(max_features=4000, ngram_range=(1, 1),
                                           sublinear_tf=True, min_df=2, max_df=0.9), 'clean_message'),
            ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                           max_features=2000, sublinear_tf=True, min_df=2), 'clean_message'),
            ('meta', StandardScaler(with_mean=False), META_COLS),
        ])),
        ('balance', SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE)),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=20,
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def screen_messages(pipe, messages: list[str]) -> pd.DataFrame:
    """Classify a batch of raw messages -> results table."""
    feats = featurize(messages)
    proba = pipe.predict_proba(feats)[:, 1]
    verdicts = ['Spam' if p >= 0.5 else 'Legitimate' for p in proba]
    return pd.DataFrame({
        'Message': messages,
        'Verdict': verdicts,
        'Spam likelihood': [f'{p:.1%}' for p in proba],
    })


def risk_signals(message: str) -> list[str]:
    """Human-readable characteristics of the message itself (not model output)."""
    signals = []
    if re.search(r'http|www\.', message, re.I):
        signals.append('Contains a web link')
    if re.search(r'\d{5,}', message):
        signals.append('Contains a long number (e.g. phone or shortcode)')
    if re.search(r'[£$€]', message):
        signals.append('Mentions money / currency')
    if message.count('!') >= 2:
        signals.append('Repeated exclamation marks')
    upper_ratio = sum(c.isupper() for c in message) / max(len(message), 1)
    if upper_ratio > 0.3 and len(message) > 15:
        signals.append('Unusually heavy use of capital letters')
    if re.search(r'\b(free|win|winner|won|prize|claim|urgent|congrat\w*)\b', message, re.I):
        signals.append('Uses common promotional / urgency wording')
    return signals


SAMPLES = {
    'Promotional offer':   "FREE entry! Text WIN to 80085 for your chance to claim a holiday voucher!",
    'Prize notification':  "WINNER!! You have been selected to receive a £900 prize. Call 09061701461 now!",
    'Account alert':       "Urgent! Your bank account has been suspended. Verify at http://secure-login.tk now",
    'Casual chat':         "Hey, are we still meeting for lunch tomorrow at 1pm?",
    'Study request':       "Can you send me the lecture notes from today please?",
    'Running late':        "Sorry I'm running 10 mins late, traffic is terrible. See you soon x",
}

VERDICT_CARD = """
<div style="border-radius:12px;padding:1.2rem 1.5rem;margin:0.5rem 0 1rem 0;
            background:{bg};border:1px solid {border};">
  <div style="font-size:1.5rem;font-weight:700;color:{fg};">{icon} {title}</div>
  <div style="color:{fg};opacity:0.85;margin-top:0.25rem;">{subtitle}</div>
</div>
"""


def render_verdict(is_spam: bool, proba: float):
    if is_spam:
        st.markdown(VERDICT_CARD.format(
            bg='#fdecea', border='#f5b7b1', fg='#922b21', icon='🚫',
            title='Spam detected', subtitle=f'This message was flagged with {proba:.1%} spam likelihood. '
                                            'It would be moved to the spam folder.'),
            unsafe_allow_html=True)
    else:
        st.markdown(VERDICT_CARD.format(
            bg='#e9f7ef', border='#a9dfbf', fg='#1e8449', icon='✅',
            title='Legitimate message', subtitle=f'This message passed screening ({proba:.1%} spam likelihood). '
                                                 'It would be delivered normally.'),
            unsafe_allow_html=True)


def single_check_tab(pipe):
    st.subheader('Check a message')

    col_msg, col_sample = st.columns([3, 1])
    with col_sample:
        sample = st.selectbox('Insert a sample', list(SAMPLES.keys()), index=None,
                              placeholder='Sample (optional)', label_visibility='collapsed')
        if st.button('Use sample', use_container_width=True, disabled=sample is None):
            st.session_state['msg'] = SAMPLES[sample]
    with col_msg:
        message = st.text_area('Message text', key='msg', height=120,
                               placeholder='Type or paste a message here...',
                               label_visibility='collapsed')

    if st.button('Screen message', type='primary'):
        if not message or not message.strip():
            st.warning('Please enter a message to screen.')
            return
        feats = featurize([message])
        proba = float(pipe.predict_proba(feats)[0, 1])
        is_spam = proba >= 0.5
        render_verdict(is_spam, proba)

        signals = risk_signals(message)
        if signals:
            with st.expander('Message characteristics', expanded=is_spam):
                for s in signals:
                    st.markdown(f'- {s}')

        st.session_state.setdefault('history', [])
        st.session_state['history'].insert(0, {
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Message': message if len(message) <= 80 else message[:77] + '...',
            'Verdict': 'Spam' if is_spam else 'Legitimate',
            'Spam likelihood': f'{proba:.1%}',
        })
        st.session_state['history'] = st.session_state['history'][:HISTORY_LIMIT]

    history = st.session_state.get('history', [])
    if history:
        st.divider()
        head, clear = st.columns([4, 1], vertical_alignment='center')
        head.markdown('**Recent activity** (this session)')
        if clear.button('Clear', use_container_width=True):
            st.session_state['history'] = []
            st.rerun()
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def batch_tab(pipe):
    st.subheader('Bulk screening')
    st.markdown('Upload a **CSV** (choose the column containing the messages) or a '
                '**plain-text file** with one message per line.')

    upload = st.file_uploader('Upload file', type=['csv', 'txt'], label_visibility='collapsed')
    if upload is None:
        return

    if upload.name.lower().endswith('.csv'):
        try:
            raw = pd.read_csv(upload, encoding='latin-1')
        except Exception:
            st.error('Could not read this CSV file. Please check the format and try again.')
            return
        text_cols = [c for c in raw.columns if raw[c].dtype == object]
        if not text_cols:
            st.error('No text column found in this file.')
            return
        col = st.selectbox('Column containing the messages', text_cols)
        messages = raw[col].dropna().astype(str).tolist()
    else:
        content = upload.read().decode('utf-8', errors='replace')
        messages = [line.strip() for line in content.splitlines() if line.strip()]

    if not messages:
        st.warning('No messages found in this file.')
        return

    st.caption(f'{len(messages):,} message(s) ready to screen.')
    if st.button('Screen all messages', type='primary'):
        with st.spinner(f'Screening {len(messages):,} messages...'):
            results = screen_messages(pipe, messages)

        n_spam = int((results['Verdict'] == 'Spam').sum())
        c1, c2, c3 = st.columns(3)
        c1.metric('Messages screened', f'{len(results):,}')
        c2.metric('Flagged as spam', f'{n_spam:,}')
        c3.metric('Legitimate', f'{len(results) - n_spam:,}')

        st.dataframe(results, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        results.to_csv(buf, index=False)
        st.download_button('Download results (CSV)', buf.getvalue(),
                           file_name='screening_results.csv', mime='text/csv')


def main():
    st.set_page_config(page_title='SpamGuard — Message Screening',
                       page_icon='🛡️', layout='centered')
    st.markdown("""
        <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2.5rem;}
        </style>
    """, unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.error('The service could not start: required data file is missing. '
                 'Please contact the administrator.')
        st.stop()

    pipe = load_classifier()

    with st.sidebar:
        st.markdown('## 🛡️ SpamGuard')
        st.markdown('Automated screening that flags unwanted spam before it reaches the inbox.')
        st.markdown('---')
        st.markdown('**How to use**\n\n'
                    '1. Paste a message (or pick a sample) and press **Screen message**.\n'
                    '2. For many messages at once, use **Bulk screening** and upload a file.\n'
                    '3. Download bulk results as CSV for your records.')
        st.markdown('---')
        st.caption('Messages are processed locally and are not stored after the session ends.')

    st.title('🛡️ SpamGuard')
    st.markdown('##### Message screening for spam protection')

    tab_single, tab_batch = st.tabs(['✉️ Single message', '📂 Bulk screening'])
    with tab_single:
        single_check_tab(pipe)
    with tab_batch:
        batch_tab(pipe)

    st.divider()
    st.caption('SpamGuard · Automated message screening')


if __name__ == '__main__':
    main()


