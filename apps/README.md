## in your venv

```bash
pip install -r apps/requirements.txt
export SOLR_URL=http://localhost:8983/solr
export SOLR_CORE=trustees
export VECTOR_FIELD=vector
export MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

streamlit run apps/semantic_ui.py
```
