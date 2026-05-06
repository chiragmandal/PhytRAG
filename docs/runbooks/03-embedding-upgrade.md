# Runbook 03: Upgrading the Embedding Model

**When to use this**: When switching to a different SentenceTransformer model
(for example, from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` for higher quality).

**Key constraint**: The embedding model and the indexed vectors must always match.
Mixing models causes silently wrong retrieval results (no error, just bad answers).
This is one of the most common production bugs in RAG systems.

---

## Upgrade procedure

**Step 1**: Run the evaluation baseline on the current model.

```bash
make eval
# Note the Hit@5 and MRR@5 scores from the MLflow output.
```

**Step 2**: Update the model name in `docker-compose.yml` and your local `.env`.

```yaml
# docker-compose.yml
environment:
  EMBED_MODEL: all-mpnet-base-v2   # was all-MiniLM-L6-v2
```

Note: `all-mpnet-base-v2` produces 768-dimensional vectors. Update `VECTOR_SIZE` too:

```yaml
  VECTOR_SIZE: 768   # was 384
```

**Step 3**: Clear the existing collection (vectors are incompatible with the new model).

```bash
# Drop the existing collection in Qdrant
curl -X DELETE http://localhost:6333/collections/phytrag
```

Or use `make clean` to wipe all volumes (more thorough but also wipes MLflow runs).

**Step 4**: Rebuild the API image (new model is baked in at build time).

```bash
docker compose build api
docker compose up -d api
```

**Step 5**: Re-ingest with the new model.

```bash
make ingest
```

**Step 6**: Run the evaluation again and compare to your baseline.

```bash
make eval
# Open http://localhost:5000 to compare the two MLflow runs side by side.
```

Only promote the new model to production if the new eval scores are equal or better.

---

## Rollback

If the new model performs worse:
1. Revert `EMBED_MODEL` and `VECTOR_SIZE` in `docker-compose.yml`.
2. Delete the collection: `curl -X DELETE http://localhost:6333/collections/phytrag`
3. Rebuild and re-ingest with the original model.
4. Verify eval scores match the baseline.
