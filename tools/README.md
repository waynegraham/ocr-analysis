python tools/eval_retrieval.py \
  --ground-truth annotations/ground_truth.jsonl \
  --solr-url http://localhost:8983/solr \
  --core trustees \
  --mode bm25 \
  --k 10 \
  --out-csv runs/eval_bm25.csv

python tools/eval_retrieval.py \
  --ground-truth annotations/ground_truth.jsonl \
  --solr-url http://localhost:8983/solr \
  --core trustees \
  --mode knn \
  --k 10 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out-csv runs/eval_knn.csv

python tools/eval_retrieval.py \
  --ground-truth annotations/ground_truth.jsonl \
  --solr-url http://localhost:8983/solr \
  --core trustees \
  --mode hybrid \
  --k 10 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out-csv runs/eval_hybrid.csv

python tools/eval_retrieval.py \
  --ground-truth annotations/ground_truth.jsonl \
  --solr-url http://localhost:8983/solr \
  --core trustees \
  --mode all \
  --k 10 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out-csv runs/eval_all.csv \
  --verbose