# Cross-encoder reranking walkthrough

1. Hybrid search still gets candidates from PostgreSQL keyword and vector retrieval.
2. Reciprocal Rank Fusion combines those candidate lists first.
3. Reranking is disabled by default with `NOVA_RERANK_ENABLED=false`.
4. When enabled, the best 50 fused candidates go to the cross-encoder.
5. Each model input pairs the query with a candidate title and chunk text.
6. FastEmbed runs `Xenova/ms-marco-MiniLM-L-6-v2` locally through ONNX.
7. The candidates are sorted by cross-encoder score and then limited for the API.
8. The response score is the cross-encoder score when reranking is on.
9. Cache keys include the active reranker model so rankings cannot collide.
10. Evaluation compares top-1 and recall@3 on the existing paraphrase fixtures.

### Likely interview questions

**Q: Why keep RRF if a cross-encoder reranks the results?** A: RRF cheaply combines two retrieval methods and gives the slower model a smaller candidate pool.
**Q: What happens when the feature is off?** A: Search returns the same RRF order and never loads the reranker model.
**Q: Does the reranker replace embeddings?** A: No; embeddings retrieve candidates, then the cross-encoder scores each query-candidate pair jointly.
