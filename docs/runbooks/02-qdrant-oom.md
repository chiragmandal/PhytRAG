# Runbook 02: Qdrant Out of Memory or Unhealthy

**Trigger**: `GET http://localhost:6333/readyz` returns non-200 or the container exits.

**Owner**: On-call MLOps engineer

---

## Immediate response

**Step 1**: Check if the container is running.

```bash
docker compose ps qdrant
# If "Exit" or "Restarting": check logs
docker compose logs qdrant --tail=50
```

**Step 2**: Common causes.

| Log message | Cause | Fix |
|---|---|---|
| `OOM killer` / `Killed` | Container exceeded memory | Increase Docker Desktop memory in Settings > Resources |
| `No space left on device` | Disk full | Run `docker system prune` to remove unused layers |
| `Permission denied` | Volume mount issue | Check `qdrant_storage` volume permissions |
| Port conflict | Another process on 6333 | `lsof -i :6333`, kill conflicting process |

**Step 3**: Restart Qdrant.

```bash
docker compose restart qdrant
# Wait for healthy
sleep 10
curl http://localhost:6333/readyz
```

---

## If data is corrupted

Qdrant's WAL (write-ahead log) handles most crash scenarios. If the collection is missing after restart:

```bash
# Check what collections exist
curl http://localhost:6333/collections

# If phytrag collection is gone, re-ingest
make ingest
```

Re-ingestion takes ~8 minutes. It is safe to run at any time.

---

## Prevention

- Ensure Docker Desktop has at least 4 GB memory allocated (Settings > Resources).
- Monitor disk usage: the Qdrant volume grows at roughly 1 MB per 500 vectors.
- Do not run `make ingest` more than once without running `make clean` first (avoids duplicate vectors).
