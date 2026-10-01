"""Manuscript block: BERTopic validation of the LLM-guided taxonomy."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PrivacyConfig, TopicModelConfig
from .privacy import aggregate_log, assert_columns, require_text_allowed
from .taxonomy import build_seed_topic_list


def cosine_similarity_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return row-wise cosine similarities between two embedding matrices."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / (np.linalg.norm(first, axis=1, keepdims=True) + 1e-12)
    second = second / (np.linalg.norm(second, axis=1, keepdims=True) + 1e-12)
    return first @ second.T


def compute_embeddings(
    texts: List[str],
    topic_config: TopicModelConfig,
    privacy_config: PrivacyConfig,
) -> Tuple[Any, np.ndarray]:
    """Encode text in memory; no message-level output is written."""
    require_text_allowed(privacy_config)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Install the optional 'topic' dependencies to compute embeddings."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(topic_config.embedding_model, device=device)
    embeddings = embedder.encode(
        list(texts),
        batch_size=topic_config.embedding_batch_size,
        show_progress_bar=not privacy_config.safe_mode,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embedder, np.asarray(embeddings)


def fit_guided_bertopic(
    frame: pd.DataFrame,
    seed_topic_list: List[List[str]],
    embedder: Any,
    embeddings: np.ndarray,
    topic_config: TopicModelConfig,
    privacy_config: PrivacyConfig,
) -> Tuple[Any, List[int], np.ndarray, pd.DataFrame]:
    """Fit the guided BERTopic model using the manuscript parameters."""
    require_text_allowed(privacy_config)
    assert_columns(
        frame,
        [topic_config.id_column, topic_config.text_column],
        "message frame",
    )
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as error:
        raise ImportError(
            "Install the optional 'topic' dependencies to run BERTopic validation."
        ) from error

    model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(
            n_neighbors=topic_config.n_neighbors,
            n_components=topic_config.n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=topic_config.random_state,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=topic_config.min_cluster_size,
            min_samples=topic_config.min_samples,
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        ),
        representation_model=KeyBERTInspired(),
        seed_topic_list=seed_topic_list,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=False,
        low_memory=True,
    )

    topics, probabilities = model.fit_transform(
        frame[topic_config.text_column].astype(str).tolist(),
        embeddings=embeddings,
    )
    topic_info = model.get_topic_info()
    aggregate_log(
        "[BERTopic] "
        f"documents={len(frame):,}; "
        f"topics={(topic_info['Topic'] != -1).sum():,}; "
        f"outliers={(topic_info['Topic'] == -1).sum():,}"
    )
    return model, topics, probabilities, topic_info


def align_topics_to_ontology(
    model: Any,
    embedder: Any,
    ontology_labels: List[str],
) -> pd.DataFrame:
    """Align BERTopic clusters to ontology MAIN labels by cosine similarity."""
    information = model.get_topic_info().copy()
    information = information[information["Topic"] != -1].reset_index(drop=True)
    topic_ids = information["Topic"].tolist()

    topic_vectors = None
    if getattr(model, "topic_embeddings_", None) is not None:
        ordered_ids = [topic for topic in model.get_topics() if topic != -1]
        if len(ordered_ids) == len(model.topic_embeddings_):
            vector_lookup = {
                topic: model.topic_embeddings_[index]
                for index, topic in enumerate(ordered_ids)
            }
            if all(topic in vector_lookup for topic in topic_ids):
                topic_vectors = np.vstack([vector_lookup[topic] for topic in topic_ids])

    if topic_vectors is None:
        topic_vectors = np.asarray(
            embedder.encode(
                information["Name"].astype(str).tolist(),
                normalize_embeddings=True,
            )
        )

    ontology_vectors = np.asarray(
        embedder.encode(ontology_labels, normalize_embeddings=True)
    )
    similarities = cosine_similarity_matrix(topic_vectors, ontology_vectors)
    best_indices = similarities.argmax(axis=1)

    return pd.DataFrame(
        {
            "topic_id": topic_ids,
            "topic_size": information["Count"].tolist(),
            "main_label": [ontology_labels[index] for index in best_indices],
            "alignment_score": similarities.max(axis=1),
        }
    ).sort_values(
        ["main_label", "alignment_score", "topic_size"],
        ascending=[True, False, False],
    )


def run_bertopic_validation(
    messages: pd.DataFrame,
    seed_taxonomy: pd.DataFrame,
    topic_config: TopicModelConfig,
    privacy_config: PrivacyConfig,
) -> Dict[str, Any]:
    """Run the complete validation block and return aggregate artifacts only."""
    assert_columns(
        messages,
        [topic_config.id_column, topic_config.text_column],
        "messages",
    )
    require_text_allowed(privacy_config)
    ontology_labels, seed_topics, _ = build_seed_topic_list(seed_taxonomy)
    embedder, embeddings = compute_embeddings(
        messages[topic_config.text_column].astype(str).tolist(),
        topic_config,
        privacy_config,
    )
    model, _, _, information = fit_guided_bertopic(
        messages,
        seed_topics,
        embedder,
        embeddings,
        topic_config,
        privacy_config,
    )
    aligned = align_topics_to_ontology(model, embedder, ontology_labels)
    topic_columns = ["Topic", "Count"] if privacy_config.safe_mode else ["Topic", "Count", "Name"]
    return {
        "topic_info": information[topic_columns].copy(),
        "aligned_topics": aligned,
        "n_docs": int(len(messages)),
        "n_topics": int((information["Topic"] != -1).sum()),
    }
