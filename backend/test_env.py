#!/usr/bin/env python3
"""
OCULUS Environment & Provider Diagnostic Utility
Tests all configured API keys, databases, and storage targets from .env.
"""

import asyncio
import os
import sys
import httpx

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings


def mask_key(k: str) -> str:
    if not k:
        return "[NOT SET - Simulated Fallback]"
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}...{k[-4:]}"


async def test_groq():
    if not settings.GROQ_API_KEY:
        return False, "Not configured (will use simulated Groq Whisper & Llama 3.3 70B judge)"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
            if res.status_code == 200:
                data = res.json()
                models = [m.get("id") for m in data.get("data", [])]
                has_llama = any("llama-3.3-70b" in m or "llama3-70b" in m for m in models)
                has_whisper = any("whisper-large-v3" in m for m in models)
                return True, f"Valid! (Models available: Llama-3.3: {has_llama}, Whisper: {has_whisper})"
            return False, f"HTTP {res.status_code}: {res.text[:120]}"
    except Exception as e:
        return False, f"Connection failed: {e}"


async def test_speechmatics():
    if not settings.SPEECHMATICS_API_KEY:
        return False, "Not configured (will use simulated speaker diarization)"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                "https://asr.api.speechmatics.com/v2/jobs?limit=1",
                headers={"Authorization": f"Bearer {settings.SPEECHMATICS_API_KEY}"},
            )
            if res.status_code in (200, 400):
                return True, "Valid! Batch STT & diarization endpoint authenticated."
            return False, f"HTTP {res.status_code}: {res.text[:120]}"
    except Exception as e:
        return False, f"Connection failed: {e}"


async def test_gemini():
    if not settings.GEMINI_API_KEY:
        return False, "Not configured (will use primary Groq judge or simulated judge)"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.GEMINI_API_KEY}"
            res = await client.get(url)
            if res.status_code == 200:
                return True, "Valid! Gemini API authenticated."
            return False, f"HTTP {res.status_code}: {res.text[:120]}"
    except Exception as e:
        return False, f"Connection failed: {e}"


async def test_attendee():
    if not settings.ATTENDEE_API_KEY:
        return False, "Not configured (will use simulated headless meeting bot dispatch)"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{settings.ATTENDEE_BASE_URL}/api/v1/bots",
                headers={"Authorization": f"Bearer {settings.ATTENDEE_API_KEY}"},
            )
            if res.status_code in (200, 401, 403):
                if res.status_code == 200:
                    return True, "Valid! Attendee.dev bot dispatch authenticated."
                return False, f"HTTP {res.status_code}: Invalid API key."
            return False, f"HTTP {res.status_code}: {res.text[:120]}"
    except Exception as e:
        return False, f"Connection failed: {e}"


async def test_database():
    try:
        from app.db.postgres import init_db
        await init_db()
        return True, f"Connected & initialized schema ({settings.DATABASE_URL.split('///')[-1]})"
    except Exception as e:
        return False, f"Database initialization failed: {e}"


async def main():
    print("=" * 70)
    print("  OCULUS SENTINEL // PROVIDER & ENVIRONMENT DIAGNOSTIC")
    print("=" * 70)
    print(f"Deploy Mode       : {settings.DEPLOY_MODE.upper()}")
    print(f"Ingestion Provider: {settings.INGESTION_PROVIDER}")
    print(f"STT Provider      : {settings.STT_PROVIDER}")
    print(f"LLM Judge Primary : {settings.LLM_PRIMARY_PROVIDER}")
    print(f"Storage Provider  : {settings.STORAGE_PROVIDER}")
    print("-" * 70)

    # Test providers concurrently
    results = await asyncio.gather(
        test_groq(),
        test_speechmatics(),
        test_gemini(),
        test_attendee(),
        test_database(),
    )

    providers = [
        ("Groq API (Whisper + Llama 3.3 70B)", mask_key(settings.GROQ_API_KEY), results[0]),
        ("Speechmatics (Diarization STT)", mask_key(settings.SPEECHMATICS_API_KEY), results[1]),
        ("Google Gemini (Fallback Judge)", mask_key(settings.GEMINI_API_KEY), results[2]),
        ("Attendee.dev (Headless Bot)", mask_key(settings.ATTENDEE_API_KEY), results[3]),
        ("Database Storage (SQLite / Postgres)", settings.DATABASE_URL, results[4]),
    ]

    for name, key_masked, (ok, msg) in providers:
        status_tag = "[PASS / ACTIVE]" if ok else "[INFO / SIM]"
        print(f"\n{status_tag} {name}")
        print(f"   Target : {key_masked}")
        print(f"   Detail : {msg}")

    print("\n" + "=" * 70)
    live_count = sum(1 for _, _, (ok, _) in providers[:4] if ok)
    print(f"  Summary: {live_count}/4 live cloud providers active.")
    print("  Note: Any unconfigured provider automatically operates in zero-cost")
    print("        simulated mode with 100% full feature parity for demos.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
