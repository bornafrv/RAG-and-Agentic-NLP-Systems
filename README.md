# RAG and Agentic NLP Systems

A collection of retrieval-augmented generation and tool-using agent projects built with LangGraph, LanceDB, and modern LLM tooling. The repository contains a Persian legal assistant, a travel-planning agent, and a compact hands-on RAG evaluation notebook.

## Projects included

### 1. Persian legal RAG assistant

The legal assistant answers questions using Persian legal documents rather than relying only on a language model's internal knowledge.

#### Document pipeline

- Detects whether PDFs are scanned or text-based.
- Converts scanned pages to images and applies OCR preprocessing.
- Extracts and validates Persian text.
- Normalizes Persian and English digits.
- Detects article numbers and chunks documents by legal article.
- Adds metadata such as source file, article number, and legal domain.

#### Retrieval and generation pipeline

- Creates dense embeddings for document chunks and queries.
- Stores vectors and metadata in LanceDB.
- Rewrites ambiguous user questions.
- Classifies intent and extracts metadata filters.
- Retrieves and reranks relevant legal passages.
- Generates answers grounded in the retrieved context.
- Tracks retries and execution time through a LangGraph state machine.

#### Evaluation and interface

- Evaluates retrieval and generation with RAGAS.
- Includes a Chainlit-based user interface in `app.py`.
- Supports focused experiments through `Legal_RAG_Assistant.ipynb`.

### 2. Multi-tool travel agent

The travel agent coordinates external services and local retrieval tools to answer multi-step travel requests.

Implemented capabilities include:

- Airport and city IATA-code resolution.
- Flight search and itinerary normalization.
- Hotel search and offer selection.
- Restaurant discovery and page-content extraction.
- Weather lookup with clothing recommendations.
- Currency conversion.
- FAQ retrieval through vector search.
- Travel-book semantic search.
- Multi-step trip planning with LangGraph.
- Command-line interaction and scenario testing.

Saved notebook outputs demonstrate flight, hotel, weather, FAQ, and travel-book queries. External results depend on API availability and the date of execution.

### 3. Hands-on RAG experiments

`NLP_CA5_Q1_HandsOn.ipynb` introduces vLLM, LanceDB, LangGraph, and RAGAS through smaller experiments. One saved evaluation run reports an answer-relevancy score of approximately `0.917`.

## Repository structure

```text
.
├── Legal_RAG_Assistant.ipynb  # Persian legal document RAG pipeline
├── Travel_Agent.ipynb         # Multi-tool travel agent
├── NLP_CA5_Q1_HandsOn.ipynb   # Focused RAG and evaluation exercises
├── app.py                     # Chainlit legal-assistant application
├── chainlit.md                # Chainlit welcome content
├── FAQ.js                     # Travel FAQ knowledge source
├── requirements.txt           # Original travel-agent dependencies
├── .gitignore
└── README.md
```

## Requirements

The combined projects use several libraries, including:

- LangChain and LangGraph
- LanceDB
- OpenAI-compatible model clients
- FastEmbed
- RAGAS
- Chainlit
- PyMuPDF and OCR tooling
- pandas and PyArrow
- Travel and search APIs used by the travel agent

Because each notebook targets a slightly different runtime, follow its installation cells instead of installing every optional package globally.

## Configuration

Create a local `.env` file only when needed. Depending on the notebook, you may need keys for LLM, embedding, travel, weather, or search providers.

```env
# Example placeholders — never commit real values
OPENAI_API_KEY=replace-me
```

The `.env` file is ignored by Git. Review each notebook for the exact variables required by the selected workflow.

## Running the notebooks

1. Create a virtual environment or use a GPU-enabled hosted notebook.
2. Install the dependencies listed in the relevant notebook.
3. Configure required environment variables.
4. Provide the source documents or allow the notebook to download its public datasets.
5. Run ingestion before retrieval, and retrieval before agent evaluation.

To start the Chainlit application after preparing the legal vector database:

```bash
chainlit run app.py
```

## Data and security notes

Source PDFs, generated LanceDB databases, downloaded datasets, videos, caches, and API credentials are excluded. These artifacts are either reproducible, private, or unnecessarily large for source control.

## Technologies

`RAG` · `LangGraph` · `LanceDB` · `RAGAS` · `Chainlit` · `OCR` · `Tool-Using Agents` · `Persian NLP`
