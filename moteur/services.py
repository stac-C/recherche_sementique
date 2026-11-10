"""Cœur du TP : embeddings (API Ollama), recherche cosinus, évaluation."""
import time

import numpy as np
import requests
from django.conf import settings

from .models import Document, EvalQuestion

def _ollama_headers():
    if settings.OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return {}

#qd
# def embed(texts):
#     """Appelle l'API Ollama et renvoie une matrice (n, d) L2-normalisée."""
#     out = []
#     for i in range(0, len(texts), 16):
#         batch = texts[i:i + 16]
#         r = requests.post(f"{settings.OLLAMA_URL}/api/embed",
#                           json={"model": settings.EMBED_MODEL, "input": batch},
#                           headers=_ollama_headers(), timeout=settings.OLLAMA_TIMEOUT)
#         if r.status_code == 404:
#             for text in batch:
#                 legacy = requests.post(
#                     f"{settings.OLLAMA_URL}/api/embeddings",
#                     json={"model": settings.EMBED_MODEL, "prompt": text},
#                     headers=_ollama_headers(), timeout=settings.OLLAMA_TIMEOUT,
#                 )
#                 legacy.raise_for_status()
#                 out.append(legacy.json()["embedding"])
#             continue
#         r.raise_for_status()
#         out += r.json()["embeddings"]
#     m = np.array(out, dtype="float32")
#     return m / np.linalg.norm(m, axis=1, keepdims=True)  # produit scalaire == cosinus

def _err(r):
    try:
        return r.json().get("error", r.text)
    except ValueError:
        return r.text[:200]


def embed(texts):
    """Appelle l'API Ollama et renvoie une matrice (n, d) L2-normalisée."""
    model = settings.EMBED_MODEL
    out = []
    for i in range(0, len(texts), 16):
        batch = [t if t.strip() else " " for t in texts[i:i + 16]]
        try:
            r = requests.post(f"{settings.OLLAMA_URL}/api/embed",
                              json={"model": model, "input": batch},
                              headers=_ollama_headers(), timeout=settings.OLLAMA_TIMEOUT)
        except requests.ConnectionError:
            raise RuntimeError(f"Serveur Ollama injoignable ({settings.OLLAMA_URL})")

        if r.status_code == 404 and "model" in _err(r).lower():
            raise RuntimeError(f"Modèle « {model} » absent du serveur : lance `ollama pull {model}`")
        if r.status_code == 404:  # ancien Ollama sans /api/embed
            for text in batch:
                legacy = requests.post(f"{settings.OLLAMA_URL}/api/embeddings",
                                       json={"model": model, "prompt": text},
                                       headers=_ollama_headers(), timeout=settings.OLLAMA_TIMEOUT)
                if not legacy.ok:
                    raise RuntimeError(f"Ollama {legacy.status_code} : {_err(legacy)}")
                out.append(legacy.json()["embedding"])
            continue
        if not r.ok:
            raise RuntimeError(f"Ollama {r.status_code} : {_err(r)}")
        out += r.json()["embeddings"]

    m = np.array(out, dtype="float32")
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)

def ollama_ok():
    try:
        return requests.get(f"{settings.OLLAMA_URL}/api/tags", headers=_ollama_headers(),
                            timeout=settings.OLLAMA_HEALTH_TIMEOUT).ok
    except requests.RequestException:
        return False


def build_index():
    docs = list(Document.objects.all())
    if not docs:
        raise ValueError("Le corpus est vide.")
    t0 = time.time()
    M = embed([d.text for d in docs])
    for d, v in zip(docs, M):
        d.embedding = v.tobytes()
        d.embed_model = settings.EMBED_MODEL
    Document.objects.bulk_update(docs, ["embedding", "embed_model"])
    return {"n": len(docs), "dim": M.shape[1], "seconds": round(time.time() - t0, 2)}


def load_matrix():
    docs = list(Document.objects.exclude(embedding=None))
    if not docs:
        return [], None
    return docs, np.vstack([np.frombuffer(bytes(d.embedding), dtype="float32") for d in docs])


def search(query, k=5):
    docs, M = load_matrix()
    if M is None:
        return []
    scores = M @ embed([query])[0]
    return [(docs[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


def diagnose(rank, s_top, s_exp):
    if rank > 3 and s_exp < 0.4:
        return "Écart sémantique (paraphrase / vocabulaire éloigné)"
    if s_top - s_exp < 0.05:
        return "Confusion entre documents voisins (quasi-égalité)"
    return "Document concurrent plus similaire (ambiguïté question/corpus)"


def evaluate():
    docs, M = load_matrix()
    qs = list(EvalQuestion.objects.select_related("expected"))
    if M is None or not qs:
        raise ValueError("Il faut un index et au moins une question de test.")
    if Document.objects.filter(embedding=None).exists():
        raise ValueError("Des documents ne sont pas indexés : régénère l'index.")
    S = embed([q.question for q in qs]) @ M.T
    ids = np.array([d.id for d in docs])
    by_id = {d.id: d for d in docs}
    rows = []
    for q, s in zip(qs, S):
        order = np.argsort(-s)
        rank = int(np.where(ids[order] == q.expected_id)[0][0]) + 1
        top = by_id[ids[order[0]]]
        s_exp = float(s[np.where(ids == q.expected_id)[0][0]])
        row = {"question": q.question, "expected": q.expected, "top1": top, "rank": rank,
               "s_top": round(float(s[order[0]]), 3), "s_exp": round(s_exp, 3), "diag": ""}
        if rank > 1:
            row["diag"] = diagnose(rank, row["s_top"], row["s_exp"])
        rows.append(row)
    ranks = np.array([r["rank"] for r in rows])
    metrics = {"n": len(rows), "hit1": float((ranks == 1).mean()), "hit3": float((ranks <= 3).mean()),
               "mrr": float((1 / ranks).mean())}
    return rows, metrics
