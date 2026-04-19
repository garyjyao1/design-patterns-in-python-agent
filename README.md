# design-patterns-in-python-agent

Retrieval-based question answering agent built from the LangChain retrieval-agent template pattern.

## What it does

- Loads and chunks documents from [`garyjyao/design-patterns-in-python`](https://github.com/garyjyao/design-patterns-in-python)
- Retrieves relevant chunks using a lexical retriever
- Answers questions with an LLM when `OPENAI_API_KEY` is available, otherwise returns extractive snippets

## Run locally

```bash
python -m pip install -e .[dev]
python - <<'PY'
from retrieval_graph.graph import graph
result = graph.invoke({"question": "What is the Singleton pattern?"})
print(result["answer"])
print(result["sources"])
PY
```

## LangGraph config

The graph entrypoint is configured in `langgraph.json` as:

- `retrieval_graph`: `./src/retrieval_graph/graph.py:graph`
