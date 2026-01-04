# MASK DeepAgents

A wrapper that uses [DeepAgents SDK](https://github.com/langchain-ai/deepagents) internally while exposing a MASK A2A interface.

## Features

- **DeepAgents SDK Integration**: Uses LangChain's DeepAgents for advanced agent capabilities
- **A2A Protocol Support**: Exposes services via A2A for multi-agent orchestration
- **MASK Compatibility**: Works seamlessly with other MASK-based agents
- **Dual Tracing**: Phoenix + Langfuse observability

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              MASK DeepAgents                        │
│  ┌─────────────────────────────────────────┐       │
│  │            A2A Server                    │       │
│  │          (Port 10030)                    │       │
│  └─────────────────┬───────────────────────┘       │
│                    │                                │
│  ┌─────────────────▼───────────────────────┐       │
│  │      DeepAgentWrapper (MASK Interface)   │       │
│  │          ↓                               │       │
│  │      DeepAgents SDK                      │       │
│  └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Anthropic API Key**

### Installation

```bash
# Clone the repository
git clone https://github.com/colinlee0924/mask-deepagents.git
cd mask-deepagents

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Edit `.env` file:

```bash
# LLM Provider
ANTHROPIC_API_KEY=your-anthropic-key

# DeepAgents Configuration
DEEPAGENT_MODEL=claude-sonnet-4-20250514

# Observability
TRACING_BACKEND=dual
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

### Running

```bash
# Start the DeepAgents wrapper
uv run mask-deepagents

# Or directly
uv run python -m mask_deepagents.main
```

The agent will start on port **10030**.

## Integration with Orchestrator

This agent can be discovered by the MASK Orchestrator:

```bash
# From orchestrator
curl -X POST http://localhost:10020/ \
  -d '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"messageId":"1","role":"user","parts":[{"kind":"text","text":"Discover agents at http://localhost:10030"}]}}}'
```

## License

MIT License - see [LICENSE](LICENSE) for details.
