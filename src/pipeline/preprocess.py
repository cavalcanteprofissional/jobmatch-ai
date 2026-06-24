import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in STOP_WORDS and len(t) > 2
    ]
    return ' '.join(tokens)


def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []
