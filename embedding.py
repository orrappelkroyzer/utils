import sys
from pathlib import Path
local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path) / "config.json")

import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from utils.bedrock_utils import embed_texts_from_series


def clean_ocr_for_embedding(text: str) -> str:
    """Normalize OCR text before embedding."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        letters = sum(char.isalpha() for char in stripped)
        if letters / max(len(stripped), 1) < 0.4:
            continue
        filtered_lines.append(stripped)

    cleaned_text = "\n".join(filtered_lines)
    cleaned_text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", cleaned_text)
    cleaned_text = cleaned_text.replace("\r", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def embed_text_series(
    text_series: pd.Series,
    embedding_model_id: str = "text-embedding-3-large",
    batch_size: int = 100,
) -> np.ndarray:
    """Embed texts from a pandas Series."""
    return embed_texts_from_series(
        text_series=text_series,
        embedding_model_id=embedding_model_id,
        batch_size=batch_size,
    )


def add_embeddings_to_dataframe(
    df: pd.DataFrame,
    text_column: str = "clean_article",
    output_column: str = "embedding",
    embedding_model_id: str = "text-embedding-3-small",
    batch_size: int = 100,
    clean_text: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with an embeddings column.

    The function embeds non-empty values from `text_column` and writes each
    embedding vector to `output_column` as a numpy array. Empty texts remain None.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in dataframe")

    output_df = df.copy()
    text_series = output_df[text_column].fillna("").astype(str)
    if clean_text:
        text_series = text_series.map(clean_ocr_for_embedding)

    non_empty_mask = text_series.str.strip() != ""
    embeddings_by_row = [None] * len(output_df)

    rows_to_embed = int(non_empty_mask.sum())
    logger.info("Embedding %s rows from column '%s'", rows_to_embed, text_column)
    if rows_to_embed > 0:
        embedded_rows = embed_text_series(
            text_series=text_series[non_empty_mask],
            embedding_model_id=embedding_model_id,
            batch_size=batch_size,
        )
        for row_index, embedding in zip(np.where(non_empty_mask.to_numpy())[0], embedded_rows):
            embeddings_by_row[row_index] = np.array(embedding)

    output_df[output_column] = embeddings_by_row
    return output_df


def reduce_embeddings_to_2d(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce dense embeddings with PCA."""
    if embeddings.shape[0] < n_components:
        raise ValueError(
            f"Number of samples ({embeddings.shape[0]}) must be >= n_components ({n_components})"
        )

    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings_2d = pca.fit_transform(embeddings)
    logger.info("PCA reduction: %sD -> %sD", embeddings.shape[1], n_components)
    logger.info("Explained variance: %.2f%%", pca.explained_variance_ratio_.sum() * 100)
    return embeddings_2d


def plot_embeddings_2d(
    embeddings_2d: np.ndarray,
    labels: Optional[pd.Series] = None,
    title: str = "2D Embedding Visualization",
):
    """Create a 2D Plotly scatter for embeddings."""
    if embeddings_2d.shape[1] != 2:
        raise ValueError(f"embeddings_2d must have 2 columns, got {embeddings_2d.shape[1]}")

    df_plot = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1]})
    if labels is not None:
        if len(labels) != len(embeddings_2d):
            raise ValueError(
                f"labels length ({len(labels)}) must match embeddings length ({len(embeddings_2d)})"
            )
        df_plot["label"] = labels.values

    if labels is not None:
        return px.scatter(
            df_plot,
            x="x",
            y="y",
            color="label",
            title=title,
            labels={"x": "PC1", "y": "PC2"},
        )

    return px.scatter(df_plot, x="x", y="y", title=title, labels={"x": "PC1", "y": "PC2"})
