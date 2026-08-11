# RAG and Agentic NLP Systems

A collection of Retrieval-Augmented Generation and LangGraph projects: a Persian legal assistant, a multi-tool travel agent, and supporting RAG evaluation experiments.

## Highlights

### Persian legal assistant

- Extracts and OCRs Persian legal documents.
- Chunks laws by article and enriches them with metadata.
- Stores embeddings in LanceDB and performs filtered retrieval and reranking.
- Uses a LangGraph workflow for query rewriting, intent classification, retrieval, and answer generation.
- Includes RAGAS-based evaluation and a Chainlit interface.

### Travel agent

- Implements flight, hotel, restaurant, weather, currency, FAQ, and trip-planning tools.
- Uses vector search and external travel-data integrations.
- Orchestrates tool calls through a LangGraph agent and provides a CLI workflow.

## Repository contents

- `Legal_RAG_Assistant.ipynb`, `app.py`, `chainlit.md`: legal RAG system.
- `Travel_Agent.ipynb`, `FAQ.js`: travel agent.
- `NLP_CA5_Q1_HandsOn.ipynb`: compact RAG and evaluation exercises.
- `requirements.txt`: original travel-agent dependencies.

Source PDFs, generated vector databases, videos, caches, and API keys are excluded. Create a local `.env` with the keys referenced in the notebooks; never commit it.

## Run

Use a Python virtual environment and follow the installation cells in each notebook. Some workflows require GPU access and external API credentials.

## Topics

`rag` · `langgraph` · `lancedb` · `agents` · `chainlit` · `ragas` · `persian-nlp`
