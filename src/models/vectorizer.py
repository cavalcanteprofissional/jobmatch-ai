from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

VECTORIZER_PATH = Path("data/models/tfidf_vectorizer.pkl")


def fit_vectorizer(
    corpus: list[str],
    max_features: int = 15_000,
    save_path: Optional[Path] = None,
) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.85,
    )

    logger.info(
        "Treinando TF-IDF com %s features em %s documentos...",
        max_features, len(corpus),
    )
    vec.fit(corpus)
    logger.info("Vocabulário final: %s termos", len(vec.vocabulary_))

    if save_path is None:
        save_path = VECTORIZER_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, save_path)
    logger.info("Vetorizador salvo em: %s", save_path)

    return vec


def load_vectorizer(path: Optional[Path] = None) -> TfidfVectorizer:
    if path is None:
        path = VECTORIZER_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Vetorizador não encontrado em {path}. Execute fit_vectorizer() primeiro."
        )
    logger.debug("Vetorizador carregado de %s", path)
    return joblib.load(path)


def transform(texts: list[str], vectorizer: TfidfVectorizer):
    return vectorizer.transform(texts)
