[README.md](https://github.com/user-attachments/files/33268352/README.md)
# NLP Document Search

Build a searchable collection of technical notes using text vectorization and document similarity.

## Tools

Python, pandas, scikit-learn

## Workflow

1. Vectorize eight sample documents with TF-IDF.
2. Rank documents using cosine similarity.
3. Return the most relevant documents for a query.
4. Check retrieval against four example queries and handle queries without a match.
5. Export documents, search examples and evaluation metrics.

## Data and method

Uses fictional technical notes. Search retrieves existing text using TF-IDF; it does not generate answers with an LLM.

## Run locally

```bash
python -m pip install -r requirements.txt
python main.py
```

For the package versions used to verify the project, install `requirements-tested.txt` instead.

## Run in Google Colab

Upload `project.ipynb` to Google Colab and select **Run all**.

## Files

- `main.py`: executable Python script.
- `project.ipynb`: notebook with the same workflow and example outputs.
- `requirements.txt`: required Python packages.
- `requirements-tested.txt`: package versions used for verification.
- `outputs/`: generated data, reports and results.

## About

This is a learning project prepared with AI assistance.
