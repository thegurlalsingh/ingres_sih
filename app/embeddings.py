import os
import numpy as np
import faiss

FAISS_INDEX_PATH = 'faiss_index.index'

# Try loading SentenceTransformer, fallback to dummy embedding
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _model = None


def compute_embedding_for_text(text: str):
    if _model:
        vec = _model.encode(text, show_progress_bar=False)
        return vec.tolist()
    
    # Fallback: simple character-based embedding
    arr = np.zeros(384, dtype=float)
    for i, ch in enumerate(text[:1000]):
        arr[i % arr.shape[0]] += ord(ch)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


# Persist/load FAISS
class FaissIndex:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = None
        self.ids = []

    def build(self, embeddings, ids):
        self.index = faiss.IndexFlatL2(self.dim)
        X = np.array(embeddings).astype('float32')
        self.index.add(X)
        self.ids = ids
        self.save()

    def save(self):
        if self.index:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            np.save('ids.npy', np.array(self.ids))

    def load(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists('ids.npy'):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.ids = np.load('ids.npy').tolist()
            return True
        return False

    def search(self, query_vec, top_k=5):
        if self.index is None:
            return [], []
        D, I = self.index.search(np.array([query_vec]).astype('float32'), top_k)
        return I[0].tolist(), D[0].tolist()


faiss_index = FaissIndex()
