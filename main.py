from pathlib import Path
import json
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
("network", "Network troubleshooting", "A network connection timeout can result from packet loss. Check latency, router connectivity and the server address."),
("database", "Database backups", "Create database backups before migrations. Verify restore procedures and retention policies."),
("performance", "Frame performance", "High frame time reduces rendering performance. Profile CPU and GPU workloads to investigate frame drops."),
("security", "Account security", "Enable multifactor authentication. Do not share passwords or access tokens."),
("data", "Missing data", "Missing numerical sensor values can be imputed with a median. Report missing counts before cleaning."),
("deployment", "Application deployment", "Test the application build before deployment. Record versions and prepare a rollback procedure."),
("logging", "Session logging", "Export simulation session logs as CSV with a session identifier, timestamp and numerical measurements."),
("model", "Model evaluation", "Split training and test data before fitting a model. Compare validation metrics against a baseline.")]
frame = pd.DataFrame(docs, columns=["id", "title", "text"])
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(frame.text)
def search(query, top_k=3):
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    positions = scores.argsort()[::-1][:top_k]
    return [{"id": frame.iloc[i].id, "title": frame.iloc[i].title, "score": round(float(scores[i]), 4)} for i in positions if scores[i] >= 0.05]
tests = [("packet loss and network timeout", "network"), ("median for missing sensor values", "data"), ("training and test split", "model"), ("CPU GPU frame drops", "performance")]
results = [{"query": q, "expected": expected, "matches": search(q)} for q, expected in tests]
assert search("zzzxxyy") == []
correct = sum(bool(r["matches"]) and r["matches"][0]["id"] == r["expected"] for r in results)
metrics = {"documents": len(frame), "test_queries": len(tests), "top1_accuracy_on_tiny_demo": correct / len(tests), "unknown_query_abstains": True}
frame.to_csv(OUT / "fictional_documents.csv", index=False)
(OUT / "retrieval_examples.json").write_text(json.dumps(results, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
