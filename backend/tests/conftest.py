import atexit
import os
import shutil
import tempfile
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

_TEST_DB_DIR = tempfile.mkdtemp(prefix="oculus_pytest_db_")
_TEST_DB_PATH = Path(_TEST_DB_DIR) / "test_oculus.db"

# Force isolated temp SQLite DB and offline keyless defaults before importing app.config/postgres.
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB_PATH.as_posix()}"
for _secret_var in (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "SPEECHMATICS_API_KEY",
    "ATTENDEE_API_KEY",
):
    os.environ[_secret_var] = ""

import app.db.postgres as _pg  # noqa: E402

# NullPool closes aiosqlite worker threads immediately on session exit, avoiding shutdown hangs on Windows.
_pg.engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool, future=True)
_pg.async_session_factory = async_sessionmaker(
    bind=_pg.engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


atexit.register(shutil.rmtree, _TEST_DB_DIR, ignore_errors=True)
