from functools import lru_cache
from typing import Iterable, List, Optional, Union

import torch
from nltk.tokenize.api import StringTokenizer
from numpy import ndarray
from transformers import AutoModel, AutoTokenizer

from mitoolspro.utils.functions import iterable_chunks


@lru_cache(maxsize=1)
def get_specter_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return AutoModel.from_pretrained("allenai/specter").to(device)


@lru_cache(maxsize=1)
def get_specter_tokenizer():
    return AutoTokenizer.from_pretrained("allenai/specter")


def huggingface_specter_embed_texts(
    texts: Union[List[str], str], batch_size: Optional[int] = 32
) -> List[ndarray]:
    if isinstance(texts, str):
        texts = [texts]
    model = get_specter_model()
    tokenizer = get_specter_tokenizer()
    embeddings = []
    for chunk in iterable_chunks(texts, batch_size):
        embeddings.extend(huggingface_specter_embed_chunk(chunk, tokenizer, model))
    return embeddings


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
