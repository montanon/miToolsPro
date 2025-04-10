from functools import lru_cache
from typing import Iterable, List, Optional, Union

import torch
from nltk.tokenize.api import StringTokenizer
from numpy import ndarray
from pandas import DataFrame
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from mitoolspro.utils.functions import iterable_chunks


@lru_cache(maxsize=1)
def get_model(model_name: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return AutoModel.from_pretrained(model_name).to(device)


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def huggingface_embed_texts(
    texts: Union[List[str], str],
    batch_size: Optional[int] = 32,
    model_name: str = "allenai/specter",
) -> DataFrame:
    if isinstance(texts, str):
        texts = [texts]
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    embeddings = []
    for chunk in iterable_chunks(texts, batch_size):
        embeddings.extend(huggingface_specter_embed_chunk(chunk, tokenizer, model))
    return DataFrame(embeddings)


def huggingface_specter_embed_chunk(
    chunk: Iterable, tokenizer: StringTokenizer, model: AutoModel
) -> List[ndarray]:
    inputs = tokenizer(
        chunk, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = inputs.to(device)
    with torch.no_grad():
        result = model(**inputs)
        return result.last_hidden_state[:, 0, :].detach().to("cpu").numpy().tolist()


def umap_embeddings(
    embeddings: DataFrame,
    n_neighbors: int = 15,
    n_components: int = 2,
    metric: str = "euclidean",
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    min_dist: float = 0.1,
    random_state: Optional[int] = None,
    output_columns: Optional[List[str]] = None,
) -> DataFrame:
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)

    if output_columns is None:
        output_columns = [f"umap_{i}" for i in range(n_components)]

    return DataFrame(reduced, columns=output_columns, index=embeddings.index)
