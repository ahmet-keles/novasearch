# NovaSearch metrics walkthrough

1. The API collects operational measurements in a thread-safe, in-process `Metrics` object.
2. Middleware times completed HTTP requests and keeps only the latest 1,000 samples.
3. The nearest-rank 95th percentile is calculated from that bounded latency window.
4. Requests to `/stats` are excluded so polling does not change the latency it reports.
5. `SearchCache` records each cache lookup as a hit or miss, including Redis failures as misses.
6. The hit rate is hits divided by all recorded search-cache lookups.
7. Ingestion and semantic search use a helper that times each embedding-provider call.
8. Embedding measurements include attempted calls, number of texts, and cumulative elapsed time.
9. `GET /stats` returns the latency, cache, and embedding measurements as JSON.
10. The counters are process-local, reset on restart, and are not shared across API workers.
11. Unit tests cover percentile, hit-rate, embedding totals, and failed embedding attempts.
12. An integration test checks the endpoint after document ingestion and cached searches.
13. **Q: Why keep only 1,000 request times?** A: Memory stays bounded while the percentile reflects recent traffic.
14. **Q: Does the cache hit rate include health or document requests?** A: No, it counts search-cache lookups only.
15. **Q: Is embedding time billing cost?** A: No, it measures local work time and calls; there is no hosted-model charge estimate.
