"""Tests for mask-deepagents."""

import asyncio
import logging
from uuid import uuid4

import httpx


async def test_deepagent():
    """Test the DeepAgents wrapper A2A endpoint."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = "http://localhost:10030"

    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info("Testing mask-deepagents...")

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What is 2 + 2?",
                        }
                    ],
                }
            },
        }

        try:
            response = await client.post(base_url, json=request)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response: {result}")
            logger.info("Test passed!")
        except Exception as e:
            logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_deepagent())
