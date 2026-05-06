# Runbook 01: Query Latency p99 Spike

**Trigger**: Grafana alert fires when `phytrag_query_latency_seconds` p99 exceeds 30 seconds for more than 2 minutes.

**Owner**: On-call MLOps engineer
**Escalation**: If not resolved in 30 minutes, escalate to ML team (LLM issue) or DevOps (infra issue)

---

## Triage checklist (first 5 minutes)

**Step 1**: Check which layer is slow.

```bash
# Open Grafana and look at the two panels side by side:
# - "Query Latency Percentiles"   (end-to-end)
# - "Time to First Token (TTFT)"  (LLM only)
```

If **TTFT is high but total latency is close to TTFT**: the bottleneck is the LLM (Ollama). Go to section A.
If **TTFT is normal but total latency is high**: the bottleneck is retrieval or the API itself. Go to section B.
If **both are normal but the alert is firing**: check if the alert is a false positive from a single slow query. Go to section C.

---

## Section A: LLM (Ollama) is slow

**Check 1**: Is Ollama still running?

```bash
curl http://localhost:11434/api/tags
# Expected: JSON list of models
# If unreachable: restart Ollama
ollama serve
```

**Check 2**: Is the GPU under memory pressure?

```bash
# macOS
sudo powermetrics --samplers gpu_power -n 1
# Look for GPU memory pressure in Activity Monitor > GPU History
```

**Check 3**: Is another process consuming the LLM?

```bash
ps aux | grep ollama
# If multiple processes: kill the stale ones and restart
```

**Mitigation**: If Ollama is healthy but slow, reduce `LLM_MAX_TOKENS` in `docker-compose.yml` to 256 temporarily. Restart the API service:

```bash
docker compose restart api
```

---

## Section B: Retrieval or API is slow

**Check 1**: Is Qdrant healthy?

```bash
curl http://localhost:6333/readyz
# Expected: {"result":true}

curl http://localhost:6333/collections/phytrag
# Check status field: should be "green"
```

**Check 2**: Is the Qdrant collection abnormally large?

```bash
curl http://localhost:6333/collections/phytrag | python3 -m json.tool | grep vectors_count
```

If `vectors_count` is unexpectedly large (suggests accidental re-ingestion), see runbook 02.

**Check 3**: Check API logs for slow embed calls.

```bash
docker compose logs api --tail=100 | grep "query_complete"
# Look for latency_ms values and where time is being spent
```

---

## Section C: Investigate slow outliers

Single slow queries (p99 vs p50 divergence) are expected with LLMs. Check:

```bash
# Is the p50 also elevated? If not, it is tail latency, not a systemic issue.
# Open Prometheus and run:
# histogram_quantile(0.50, rate(phytrag_query_latency_seconds_bucket[5m]))
# histogram_quantile(0.99, rate(phytrag_query_latency_seconds_bucket[5m]))
```

If p50 is normal and p99 is elevated, this is acceptable latency variance for an LLM service. Consider widening the alert threshold.

---

## Post-incident

1. Add a note to this runbook describing the root cause and resolution.
2. If the issue was reproducible, add a test case to `tests/test_api.py`.
3. Update the alert threshold if it was a false positive.
