# arXiv Papers Explainer — Full Project Guide

## What This Project Does

This is a Python CLI tool that automates the entire research literature review workflow. You give it a research topic (like *"attention mechanisms in transformers"*), and it:

1. **Searches** arXiv for the 5 most relevant papers
2. **Downloads** each paper's PDF and extracts the full text
3. **Critiques** each paper's methodology using an LLM (DeepSeek)
4. **Synthesizes** everything into a structured literature review with an introduction, thematic grouping, methodological comparison, gaps, and conclusions

It can also do a **deep-dive** on a single paper: download it, extract the text, and generate a plain-English explanation that a high school graduate could understand — no jargon, full of analogies and everyday examples.

---

## How the Code Is Organized

```
arxiv_papers_explainer/
├── main.py                  # Entry point: CLI + pipeline runner
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py          # Public API re-exports
│   ├── state.py             # Data models (what flows through the pipeline)
│   ├── utils.py             # Shared resources (LLM, arXiv client)
│   ├── graph/
│   │   ├── __init__.py      # Re-exports build_graph
│   │   └── graph.py         # The pipeline orchestration (LangGraph)
│   ├── nodes/
│   │   ├── __init__.py      # Re-exports all node functions
│   │   ├── search_agent.py  # Node 1: Search arXiv
│   │   ├── paper_explainer.py # Node 2 (optional): Simple-language deep-dive
│   │   ├── reader.py        # Node 3: Download PDFs, extract text
│   │   ├── critique.py      # Node 4: LLM methodological critique
│   │   └── synthesis.py     # Node 5: LLM literature review synthesis
│   ├── tools/               # Reserved for future tool definitions (empty)
│   └── agents/              # Reserved for future agent definitions (empty)
└── tests/                   # Test directory (empty scaffold)
```

---

## Core Concepts (No Prior Knowledge Needed)

### 1. LangGraph — The Orchestration Engine

The project uses **LangGraph**, a library for building AI workflows as directed graphs. Think of it like a flowchart where each box (called a **node**) does one job, and arrows (**edges**) decide which box runs next.

A node is just a Python function that receives a state object, modifies it, and returns it. The same state object travels through every node, accumulating data as it goes.

### 2. LLM (Large Language Model) — The Brain

The project uses **DeepSeek** (a ChatGPT-like AI model) for three things:
- Critiquing the methodology of each paper
- Explaining papers in simple language
- Writing the final literature review

It communicates with DeepSeek through the OpenAI-compatible API protocol.

### 3. arXiv — The Paper Source

[arXiv](https://arxiv.org) is a free online repository of scientific papers. The project uses the `arxiv` Python library to search it and get paper metadata (title, authors, abstract, PDF URL).

---

## The Data Model (`src/state.py`)

The `AgentState` class is the "backpack" that carries all data through the pipeline. It's a Pydantic model (a Python data class with validation). Here's what it holds at each stage:

| Field | Type | Populated By | Purpose |
|-------|------|-------------|---------|
| `original_query` | `str` | User (CLI arg) | The research question or topic |
| `target_paper_id` | `str` | User (--paper flag) | Specific arXiv ID for deep-dive |
| `papers` | `list[PaperMetadata]` | `search_agent` | Search results with title, authors, abstract, URLs |
| `paper_texts` | `list[PaperContent]` | `reader` or `paper_explainer` | Full plain-text extracted from each PDF |
| `critiques` | `list[Critique]` | `critique` | Strengths, weaknesses, methodology notes, novelty score per paper |
| `final_draft` | `str` | `synthesis` | The final literature review document |

Supporting models:
- **`PaperMetadata`**: arxiv_id, title, authors, abstract, publication date, PDF link
- **`PaperContent`**: arxiv_id + full text string
- **`Critique`**: arxiv_id + strengths (list), weaknesses (list), methodology_notes, novelty_assessment, score (0-10)

---

## The Pipeline Flow (`src/graph/graph.py`)

The `build_graph()` function creates a **StateGraph** with 5 nodes and conditional routing. The state flows through them in this order:

```
                    ┌─────────────────┐
                    │  search_agent    │  ← Always runs first
                    │  (searches arXiv)│
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │  routing logic  │
                    └─┬────┬──────┬──┘
                      │    │      │
            ┌─────────┘    │      └──────────┐
            ▼              ▼                 ▼
   ┌──────────────┐  ┌──────────┐      ┌──────────┐
   │paper_explainer│  │  reader   │      │   END    │
   │(deep-dive on  │  │(download │      │(no papers│
   │ single paper) │  │  PDFs)   │      │ found)   │
   └──────┬───────┘  └────┬─────┘      └──────────┘
          │               │
          │        ┌──────▼──────┐
          │        │  routing:    │
          │        │  has texts?  │
          │        └──┬──────┬───┘
          │           │      │
          └───────────┘      ▼
                    │     ┌──────┐
                    │     │ END  │
                    ▼     └──────┘
            ┌──────────┐
            │ critique  │  ← LLM analyzes each paper
            └────┬─────┘
                 │
          ┌──────▼──────┐
          │ routing:     │
          │ has critiques?
          └──┬──────┬───┘
             │      │
             ▼      ▼
     ┌──────────┐ ┌──────┐
     │synthesis │ │ END  │
     │(LLM writes│└──────┘
     │ review)  │
     └────┬─────┘
          ▼
      ┌──────┐
      │ END  │
      └──────┘
```

### The Three Routing Decisions

1. **After search**: If no papers found → END. If `target_paper_id` is set → `paper_explainer`. Otherwise → `reader`.
2. **After reader**: If text was extracted → `critique`. Otherwise → END.
3. **After critique**: If critiques were generated → `synthesis`. Otherwise → END.

### Two Modes of Operation

**Full pipeline** (no `--paper` flag):
```
search → reader → critique → synthesis → final literature review
```

**Deep-dive mode** (`--paper 2301.12345`):
```
search → paper_explainer → reader → critique → synthesis
```
The `paper_explainer` downloads that specific paper and generates a simple-language explanation first, then continues the normal pipeline.

---

## Node-by-Node Walkthrough

### Node 1: `search_agent` (`src/nodes/search_agent.py`)

**What it does**: Searches arXiv for the top 5 papers matching the user's query, sorted by relevance.

**How it works**:
1. Reads `state.original_query`
2. Creates an arXiv `Search` object with `max_results=5`
3. Calls the arXiv API through the shared `arxiv_client`
4. For each result, extracts the arXiv ID (e.g., `2301.12345` from the full URL)
5. Builds a `PaperMetadata` object for each paper
6. Prints a formatted summary to the terminal
7. Sets `state.papers` and returns

**Key detail**: If the search fails or returns nothing, it quietly returns the state unchanged — downstream routing will send it to END.

---

### Node 2: `paper_explainer` (`src/nodes/paper_explainer.py`)

**What it does**: Downloads a single paper's PDF, extracts its text, and asks the LLM to explain it in plain language suitable for someone with no academic background.

**How it works**:
1. Reads `state.target_paper_id`
2. Looks up that paper in `state.papers` (or fetches it from arXiv if not found)
3. Downloads the PDF to a temporary file using `urllib`
4. Extracts text from each page using PyPDF2
5. Deletes the temp file
6. Takes the first 8,000 characters of the paper
7. Sends it to the LLM with this system prompt:
   > *"You are a research assistant that explains academic papers so a high school graduate can understand them. Avoid jargon. Use analogies and everyday examples..."*
8. Prints the explanation with the paper's title, authors, and arXiv link
9. Adds the full extracted text to `state.paper_texts` (so the rest of the pipeline can use it)

**Why 8,000 characters?** LLMs have context limits, and academic papers are often 30,000+ characters. The first 8,000 characters typically cover the abstract, introduction, and approach — enough to explain the core ideas.

---

### Node 3: `reader` (`src/nodes/reader.py`)

**What it does**: Downloads the PDF for every paper in `state.papers` and extracts the full text.

**How it works**:
1. Iterates through `state.papers`
2. Skips any paper that already has text in `state.paper_texts` (idempotent)
3. Constructs the PDF URL (from metadata or the standard arXiv format)
4. Downloads to a temp file
5. Extracts text page by page with PyPDF2
6. Deletes the temp file
7. Appends a `PaperContent` to `state.paper_texts`
8. If download or extraction fails for a paper, it silently skips it

**Why PyPDF2?** It's a pure-Python PDF reader that extracts text without external dependencies. The trade-off is it can struggle with complex layouts, equations, and figures — those just appear as garbled text or are skipped entirely.

---

### Node 4: `critique` (`src/nodes/critique.py`)

**What it does**: For each paper with extracted text, asks the LLM to evaluate its methodology and produce a structured critique.

**How it works**:
1. Iterates through `state.paper_texts`
2. For each, finds the matching `PaperMetadata` (for title, authors, abstract)
3. Takes the first 6,000 characters of the full text
4. Sends to the LLM with this prompt:
   > *"You are a senior research methodologist reviewing an academic paper. Evaluate the paper honestly and concisely. Identify what the paper does well, where it falls short, and assess its novelty. Return your evaluation as a JSON object..."*
5. Parses the LLM's JSON response into a `Critique` object with: strengths, weaknesses, methodology_notes, novelty_assessment, and a 0-10 score
6. Handles markdown-wrapped JSON (strips ``` fences)
7. Prints a summary of the critique
8. Appends to `state.critiques`

**Why JSON output?** The critique node needs structured data (lists of strengths/weaknesses, numeric score) that downstream nodes can work with programmatically. By asking the LLM to return JSON, we get machine-parseable output from what is fundamentally unstructured text generation.

**Error handling**: If the LLM returns invalid JSON, that paper is simply skipped. The pipeline doesn't crash.

---

### Node 5: `synthesis` (`src/nodes/synthesis.py`)

**What it does**: Combines all papers, their extracted text, and their critiques into a single structured literature review.

**How it works**:
1. Builds a compact text block for each paper, including:
   - arXiv ID, title, authors, abstract excerpt
   - Full text excerpt (first 1,500 characters)
   - Critique: score, strengths, weaknesses, novelty assessment
2. Sends everything to the LLM with a prompt that specifies exact sections:
   - **1. Introduction** — research area and why it matters
   - **2. Key Themes and Findings** — grouped by common themes
   - **3. Methodological Comparison** — approaches, datasets, evaluation
   - **4. Gaps and Open Problems** — what's missing, future directions
   - **5. Conclusion** — state of the field, most promising directions
3. The LLM writes an 800-1,500 word draft in academic but accessible style
4. Sets `state.final_draft` and prints it

---

## Shared Utilities (`src/utils.py`)

Two shared resources used by all nodes:

**arXiv Client** (`arxiv_client`):
```python
arxiv_client = Client(page_size=100, delay_seconds=3.0, num_retries=5)
```
- `delay_seconds=3.0` respects arXiv's rate limit (they block aggressive scrapers)
- `num_retries=5` handles transient network failures

**LLM Instance** (`llm`):
```python
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=_LLM_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)
```
- Uses the OpenAI-compatible protocol to talk to DeepSeek
- `temperature=0.3` means it's fairly deterministic (low creativity) — good for factual analysis
- Falls back to a hardcoded API key if the `DEEPSEEK_API_KEY` env var isn't set

**`extract_arxiv_id()`**: Takes a full arXiv entry URL (like `http://arxiv.org/abs/2301.12345v2`) and extracts just the ID (`2301.12345`). The `v2` suffix is a version number — this function strips it because we want the canonical paper ID.

---

## The Entry Point (`main.py`)

The `main()` function ties everything together:

1. **Parse CLI arguments**:
   - `query` — positional, the research topic
   - `--paper` — arXiv ID for deep-dive mode
   - `--skip-critique` — (declared but not yet wired into the graph)
   - `--output` — save the final draft to a file
   - `--verbose` — enable debug logging

2. **Create initial state**: An `AgentState` with just `original_query` and `target_paper_id` — everything else starts empty.

3. **Build and invoke the graph**: `graph.invoke(state)` runs the entire pipeline. The same state object flows through every node.

4. **Extract and output results**: The final state (a dict or AgentState) contains `final_draft` — print it and optionally save to a file.

---

## How the Pieces Connect (Data Flow)

Here's the full lifecycle of data through the system:

```
User types: python main.py "graph neural networks for drug discovery"

  1. main.py creates AgentState(original_query="graph neural networks for drug discovery")
                │
  2. search_agent runs ─── state.papers = [PaperMetadata(#1), ..., PaperMetadata(#5)]
                │
  3. reader runs ─── downloads 5 PDFs, extracts text
                └── state.paper_texts = [PaperContent(#1), ..., PaperContent(#5)]
                │
  4. critique runs ─── for each paper, LLM returns JSON
                └── state.critiques = [Critique(#1), ..., Critique(#5)]
                │
  5. synthesis runs ─── LLM writes review from papers + critiques
                └── state.final_draft = "## 1. Introduction\n\nGraph neural networks..."
                │
  6. main.py prints final_draft to terminal (or saves to file)
```

---

## Dependencies Explained

| Package | Why It's Needed |
|---------|----------------|
| `langgraph>=0.4.0` | Graph-based workflow orchestration — defines the pipeline |
| `langchain-core>=0.3.0` | Prompt templates and chain primitives for LLM interactions |
| `langchain-openai>=0.3.0` | OpenAI-compatible LLM client (used to talk to DeepSeek) |
| `arxiv>=2.1.0` | Official arXiv API client for searching and fetching papers |
| `PyPDF2>=3.0.0` | Extracts plain text from downloaded PDFs |
| `pydantic>=2.0.0` | Data validation for the state models |
| `httpx>=0.27.0` | HTTP client (used internally by the OpenAI client) |

---

## What Happens in Each Scenario

### Normal run: `python main.py "attention mechanisms"`
```
search → reader → critique → synthesis → prints review
```
Searches arXiv, downloads 5 papers, critiques each, writes a review.

### Deep-dive: `python main.py "attention" --paper 1706.03762`
```
search → paper_explainer → reader → critique → synthesis
```
Searches arXiv (finds the "Attention Is All You Need" paper among results), then explains it in plain English, then continues the normal pipeline.

### Save to file: `python main.py "graph neural networks" --output review.md`
Same as normal run, but writes the final draft to `review.md`.

### No results: Search returns nothing → graph routes to END immediately → prints "No literature review was generated."

---

## Design Decisions Worth Noting

1. **Idempotent nodes**: Each node checks if its work is already done (e.g., `reader` skips papers already in `paper_texts`, `critique` skips those already in `critiques`). This means the graph can be re-invoked safely.

2. **Graceful degradation**: If a PDF download fails, the paper is skipped. If the LLM returns bad JSON, that critique is skipped. The pipeline never crashes from one bad paper.

3. **Character limits**: Full papers are too long for LLM context windows, so the code sends only the first 6,000-8,000 characters. This is a pragmatic constraint — the intro and approach sections are usually the most informative anyway.

4. **Shared arXiv client**: A single `arxiv_client` instance is used everywhere, with built-in rate limiting (3-second delay between requests) to avoid being blocked.

5. **DeepSeek over OpenAI**: The project uses DeepSeek's API, which is significantly cheaper than OpenAI for equivalent model quality. The `ChatOpenAI` class works because DeepSeek provides an OpenAI-compatible endpoint.

6. **`--skip-critique` is parsed but unused**: The flag exists in the CLI parser but isn't wired into the graph routing logic yet — it's declared for future use.

---

## Docker Setup (Recommended)

```bash
# Build the image (run once after cloning)
docker build -t arxiv-explainer .

# Run with your API key
docker run --rm -e DEEPSEEK_API_KEY="your-key-here" arxiv-explainer "graph neural networks"

# Save output to a file on your machine
docker run --rm -e DEEPSEEK_API_KEY="your-key" -v "$(pwd):/app" arxiv-explainer "topic" --output review.md

# Enable verbose logging
docker run --rm -e DEEPSEEK_API_KEY="your-key" arxiv-explainer "quantum computing" --verbose
```

No Python, venvs, or packages needed on your machine — just Docker Desktop.

---

## How to Set Up and Run (Without Docker)

### Step 1: Install Python

You need Python 3.10 or newer. Check with:
```bash
python3 --version
```

### Step 2: Clone and Install Dependencies

```bash
cd arxiv_papers_explainer
python3 -m venv .venv          # Create an isolated environment
source .venv/bin/activate       # Activate it (macOS/Linux)
pip install -r requirements.txt # Install all dependencies
```

### Step 3: Set Your API Key

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

Get a key from [platform.deepseek.com](https://platform.deepseek.com). Without this, the code falls back to a hardcoded key embedded in `src/utils.py`, which may or may not work.

### Step 4: Run

```bash
# Full literature review
python main.py "transformer attention mechanisms"

# Deep-dive on a specific paper
python main.py "attention" --paper 1706.03762

# Save output to a file
python main.py "graph neural networks" --output review.md

# Enable debug logging to see every step
python main.py "federated learning" --verbose
```

---

## Technology Deep-Dive: How LangGraph Orchestrates the Pipeline

### What is a StateGraph?

A `StateGraph` is LangGraph's core abstraction. It's a directed graph where:
- **Nodes** are functions that read from and write to a shared state
- **Edges** define which node runs next
- The **state** is a single object that flows through every node

Here's the exact code that builds the graph (from `src/graph/graph.py`):

```python
graph = StateGraph(AgentState)       # 1. Create graph with AgentState as the state type

graph.add_node("search_agent", search_agent)  # 2. Register nodes
graph.add_node("paper_explainer", paper_explainer)
graph.add_node("reader", reader)
graph.add_node("critique", critique)
graph.add_node("synthesis", synthesis)

graph.set_entry_point("search_agent")         # 3. Where to start

graph.add_conditional_edges(                  # 4. Branching after search
    "search_agent",
    _route_after_search,
    {"explainer": "paper_explainer", "reader": "reader", "end": END},
)

graph.add_edge("paper_explainer", "reader")   # 5. Fixed edge
# ... more conditional edges ...

return graph.compile()                        # 6. Compile and return
```

### How Conditional Edges Work

The routing functions take the current state and return a string:

```python
def _route_after_search(state: AgentState) -> str:
    if not state.papers:
        return "end"                          # No papers → stop
    if state.target_paper_id.strip():
        return "explainer"                    # Has target → deep-dive
    return "reader"                           # Normal → read papers
```

LangGraph uses that return value to look up the next node in the mapping dictionary. This is how the graph "decides" which path to take based on the data.

### The State is Mutable

Even though `AgentState` is a Pydantic BaseModel (which is normally immutable), LangGraph nodes don't modify it in place — they return a new state. However, because the code does `state.papers = papers` and returns `state`, it looks like mutation. Under the hood, LangGraph merges the returned dict-like object into the running state.

---

## Technology Deep-Dive: How the LLM Integration Works

### The Chain Pattern

The project uses LangChain's **chain** pattern with the pipe operator:

```python
chain = _CRITIQUE_PROMPT | llm
response = chain.invoke({"title": "...", "authors": "...", ...})
```

What this does:
1. `_CRITIQUE_PROMPT` is a `ChatPromptTemplate` — it takes variables (title, authors, abstract, text) and formats them into a prompt
2. `llm` is a `ChatOpenAI` instance pointed at DeepSeek's API
3. `|` (pipe) creates a chain: prompt → LLM → response
4. `chain.invoke(...)` sends the filled-in prompt to DeepSeek and returns the generated text

### Why DeepSeek?

The project uses DeepSeek instead of OpenAI directly because:
- **Cost**: DeepSeek is ~10x cheaper per token
- **Compatibility**: DeepSeek exposes an OpenAI-compatible endpoint, so `langchain-openai`'s `ChatOpenAI` class works without modification
- **Quality**: DeepSeek's models are competitive with GPT-4 on academic reasoning tasks

The only configuration needed is changing `base_url`:
```python
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",  # This is the key line
    api_key=api_key,
)
```

### The Prompt → JSON → Parse Pattern

The critique node demonstrates a common LLM integration pattern:

1. **Ask for structured output in the prompt**: "Return your evaluation as a JSON object with exactly these keys: strengths, weaknesses..."
2. **Handle markdown wrapping**: LLMs often wrap JSON in ``` fences — the code strips them
3. **Parse defensively**: `json.loads(raw)` is wrapped in try/except — if the LLM returns malformed JSON, that paper is skipped rather than crashing the pipeline

This pattern is necessary because DeepSeek's API (like most LLM APIs) doesn't have native structured output support — you have to ask nicely and handle the mess.

---

## Pydantic Models Explained

Pydantic is a data validation library. When you define:

```python
class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
```

Pydantic automatically:
- Validates types at runtime (if you try to put an `int` in `arxiv_id`, it raises an error)
- Fills in defaults for missing fields
- Enforces constraints (`ge=0.0, le=10.0` means score must be between 0 and 10)
- Provides JSON serialization/deserialization for free

The `Field(...)` with `...` (Ellipsis) means the field is **required** — Pydantic will raise an error if it's missing. `Field(default="")` means it's optional with an empty string default.

`Optional[datetime]` is Python type-annotation shorthand for `datetime | None` — the field can be a datetime or None.

---

## How PDF Extraction Works (and Its Limitations)

The PDF pipeline has three steps:

### 1. Download
```python
urllib.request.urlretrieve(pdf_url, tmp)  # Saves PDF to a temp file
```
Vanilla `urllib` — no external download library needed. arXiv allows programmatic downloads with reasonable rate limiting (the 3-second delay in the arXiv client).

### 2. Extract Text
```python
reader = PyPDF2.PdfReader(str(pdf_path))
for page in reader.pages:
    text = page.extract_text()
```

PyPDF2 reads the PDF's internal text layer. **This is the project's biggest limitation:**
- PDFs with embedded fonts or scanned pages (images) yield no text
- Mathematical equations, tables, and figures produce garbled output or are lost entirely
- Multi-column layouts can produce interleaved text from different columns

### 3. Clean Up
```python
pdf_path.unlink(missing_ok=True)  # Delete the temp file
```
Temporary files are deleted immediately after extraction — nothing persists on disk.

---

## How to Extend the Project

### Adding a New Node

1. Create `src/nodes/your_node.py` with a function:
```python
def your_node(state: AgentState) -> AgentState:
    # do something with state
    return state
```

2. Register it in `src/nodes/__init__.py`:
```python
from src.nodes.your_node import your_node
```

3. Add it to the graph in `src/graph/graph.py`:
```python
graph.add_node("your_node", your_node)
graph.add_edge("critique", "your_node")  # or use add_conditional_edges
```

### Adding a New Field to State

Add it to `AgentState` in `src/state.py`:
```python
class AgentState(BaseModel):
    # ... existing fields ...
    new_field: str = Field(default="")
```

All nodes automatically get access to it — no other changes needed.
