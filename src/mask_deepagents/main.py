"""Main entry point for mask-deepagents A2A server."""

import os
from pathlib import Path

from dotenv import load_dotenv

from mask.a2a import MaskA2AServer

from mask_deepagents.agent import create_agent

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def main():
    """Start the A2A server."""
    # Setup tracing based on TRACING_BACKEND env var
    backend = os.environ.get("TRACING_BACKEND", "dual")

    if backend == "dual":
        from mask.observability import setup_dual_tracing

        print("Using DUAL tracing (Phoenix + Langfuse)...")
        setup_dual_tracing(project_name="mask-deepagents")
    elif backend == "phoenix":
        from mask.observability import setup_openinference_tracing

        print("Using Phoenix tracing...")
        setup_openinference_tracing(
            project_name="mask-deepagents",
            endpoint=os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
            ),
        )
    elif backend == "langfuse":
        from mask.observability import setup_langfuse_otel_tracing

        print("Using Langfuse tracing...")
        setup_langfuse_otel_tracing()
    else:
        print("No tracing configured (set TRACING_BACKEND=dual|phoenix|langfuse)")

    # Create DeepAgents wrapper
    agent = create_agent()

    # Create A2A server
    server = MaskA2AServer(
        agent=agent,
        name="mask-deepagents",
        description="DeepAgents SDK wrapped with MASK A2A protocol",
    )

    port = int(os.environ.get("PORT", "10030"))
    print(f"Starting mask-deepagents on port {port}...")
    server.run(port=port)


if __name__ == "__main__":
    main()
