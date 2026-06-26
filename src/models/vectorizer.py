from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

VECTORIZER_PATH = Path("data/models/tfidf_vectorizer.pkl")
SBERT_PATH = Path("data/models/sentence_bert.pkl")
SBERT_EMBEDDINGS_PATH = Path("data/models/sentence_embeddings.npy")


class SentenceBertVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Carregando Sentence-BERT: %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Sentence-BERT carregado")
        return self._model

    def fit(self, texts: list[str], save_path: Optional[Path] = None) -> "SentenceBertVectorizer":
        logger.info("Gerando embeddings SBERT para %s documentos...", len(texts))
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        if save_path is None:
            save_path = SBERT_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        np.save(SBERT_EMBEDDINGS_PATH, embeddings)
        logger.info("Embeddings SBERT salvos em: %s", SBERT_EMBEDDINGS_PATH)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
        return embeddings

    def load_embeddings(self) -> np.ndarray:
        if not SBERT_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Embeddings SBERT não encontrados. Execute fit() primeiro."
            )
        return np.load(SBERT_EMBEDDINGS_PATH)


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


def load_sbert_vectorizer(path: Optional[Path] = None) -> SentenceBertVectorizer:
    if path is None:
        path = SBERT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"SBERT não encontrado em {path}. Execute SentenceBertVectorizer.fit() primeiro."
        )
    return joblib.load(path)
