import os
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Attendee defaults to localhost:8000 when self-hosted
ATTENDEE_API_URL = os.getenv("ATTENDEE_API_URL", "http://localhost:8000/api/v1")

def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("ATTENDEE_API_KEY")
    if not api_key:
        logger.warning("ATTENDEE_API_KEY is not set. Attendee integration will run in dry-run mode.")
        return {}
    return {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

async def create_bot(meeting_url: str, bot_name: str) -> Dict[str, Any]:
    headers = _get_headers()
    if not headers:
        return {"id": "mock_bot_id", "status": "dry_run"}
    
    payload = {
        "meeting_url": meeting_url,
        "bot_name": bot_name,
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ATTENDEE_API_URL}/bots",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to create Attendee bot: {e}")
        return {"id": "mock_bot_id", "status": "error", "error": str(e)}

async def get_bot(bot_id: str) -> Dict[str, Any]:
    headers = _get_headers()
    if not headers or bot_id == "mock_bot_id":
        return {"id": bot_id, "status": "mocked"}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{ATTENDEE_API_URL}/bots/{bot_id}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get Attendee bot status: {e}")
        return {"id": bot_id, "status": "error", "error": str(e)}

async def get_transcript(bot_id: str) -> list[Dict[str, Any]]:
    headers = _get_headers()
    if not headers or bot_id == "mock_bot_id":
        return []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{ATTENDEE_API_URL}/bots/{bot_id}/transcript",
                headers=headers
            )
            response.raise_for_status()
            # Attendee returns a list of dictionaries directly
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get Attendee transcript: {e}")
        return []

async def leave_bot(bot_id: str) -> bool:
    headers = _get_headers()
    if not headers or bot_id == "mock_bot_id":
        return True
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ATTENDEE_API_URL}/bots/{bot_id}/leave",
                headers=headers
            )
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Failed to instruct Attendee bot to leave: {e}")
        return False

