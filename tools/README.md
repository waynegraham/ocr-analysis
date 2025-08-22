


## Evaluate Retrieval

python tools/eval_retrieval.py \
  --ground-truth annotations/ground_truth.jsonl \
  --solr-url http://localhost:8983/solr \
  --core trustees \
  --mode all \
  --k 10 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out-csv runs/eval_all.csv \
  --verbose

## TODO

- Develop 10-20 questions and update the `ground_truth.jsonl`
