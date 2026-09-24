"""Multilingual E5 prefixes and token windows; avoid silently truncating Indic text."""
from functools import lru_cache
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

class RetrievalEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.e5 = 'e5' in model_name.lower()
        self.client = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})

    def _encode(self, texts, prefix):
        if not self.e5:
            return self.client.embed_documents(texts)
        model = self.client._client
        tokenizer = model.tokenizer
        limit = max(16, min(model.max_seq_length, 512) - 16)
        pieces, groups = [], []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            start = len(pieces)
            pieces.extend(prefix + tokenizer.decode(tokens[i:i+limit], skip_special_tokens=True) for i in range(0,len(tokens),limit))
            if len(pieces) == start: pieces.append(prefix)
            groups.append((start,len(pieces)))
        if not pieces: return []
        vectors = np.asarray(self.client.embed_documents(pieces))
        result = []
        for start,end in groups:
            vector = vectors[start:end].mean(axis=0)
            result.append((vector / max(np.linalg.norm(vector), 1e-12)).tolist())
        return result

    def embed_documents(self, texts):
        return self._encode(texts, 'passage: ')

    def embed_query(self, text):
        if not self.e5: return self.client.embed_query(text)
        return self._encode([text], 'query: ')[0]

@lru_cache(maxsize=2)
def shared_embeddings(model_name):
    return RetrievalEmbeddings(model_name)
