# RAG-based NLU-to-Tool Matching System

A production-ready system for mapping user queries directly to MCP server tools using semantic similarity matching with **no LLM planning required**.

## 🎯 Key Features

- ✅ **No LLM Planning**: Direct RAG-based semantic query→tool mapping
- ✅ **High Confidence Only**: 0.85 threshold from environment configuration
- ✅ **Semantic Variable Extraction**: Context-aware, no regex garbage
- ✅ **Multi-Tool Dependency Planning**: Proper execution sequencing
- ✅ **Interactive Variable Collection**: Natural language prompts with tool context
- ✅ **62 Intents**: 30 Jira + 32 HubSpot comprehensive coverage

## 📁 Clean Project Structure

```
/Users/prithvi/Projects/poc/wit-style/
├── .env                           # Environment configuration  
├── requirements.txt               # Python dependencies
├── cli.py                         # Main CLI interface
├── README.md                      # Documentation
├── jira-intent-config.yaml        # 30 Jira intents
├── hubspot-intent-config.yaml     # 32 HubSpot intents  
├── models/                        # FAISS indexes (auto-generated)
└── src/                           # Core modules
    ├── __init__.py
    ├── config_parser.py           # YAML parser
    ├── embedding_engine.py        # FAISS + sentence-transformers
    ├── intent_matcher.py          # Main orchestrator
    ├── dependency_planner.py      # Tool sequencing
    └── lightweight_nlp.py         # Variable extraction + NLP
```

## 🚀 Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Test with interactive prompting
python3 cli.py --threshold 0.3 test "Change the status"
```

### Configuration (.env)
```bash
CONFIDENCE_THRESHOLD=0.85          # Match confidence requirement
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
INDEX_DIR=./models
```

## 💡 Perfect Demo

**Query:** `"Change the status"` (incomplete)

**System Response:**
```
🤖 I need two things:
1. Which Jira issue should I work with? (e.g., PROJ-123)
2. What status should I change it to? (e.g., In Progress, Done, QA)

What I'll do: I'll check what status transitions are available 
for this issue and then change it to your desired status

📝 User: "Change PROJ-789 to Done"
✅ Understood: {'issue_key': 'PROJ-789', 'status': 'Done'}

Multi-Tool Plan:
get_issue_transitions → transition_issue
```

## 🛠️ MCP Integration

**MCP Servers (Production-Ready):**
- Jira MCP Server: `/Users/prithvi/Documents/Cline/MCP/jira-server/`
- HubSpot MCP Server: `/Users/prithvi/Documents/Cline/MCP/hubspot-server/`

Both configured in Cline + Claude Desktop with real API credentials.

---

**🎯 Complete production system that eliminates LLM planning while providing intelligent semantic matching and context-aware variable collection.**
