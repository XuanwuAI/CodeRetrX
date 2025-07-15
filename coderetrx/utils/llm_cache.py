import hashlib
import sqlite3
import json
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
from coderetrx.utils.path import get_cache_dir
from pydantic_settings import BaseSettings, SettingsConfigDict
from logging import getLogger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = getLogger(__name__)


class SqliteCacheSettings(BaseSettings):
    """SQLite cache provider settings."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )
    db_path: str = Field(
        default_factory=lambda: str(get_cache_dir() / "llm" / "cache.db"),
        alias="SQLITE_DB_PATH",
        description="Path to SQLite database file",
    )


class RedisCacheSettings(BaseSettings):
    """Redis cache provider settings."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8", env_file=".env", extra="allow"
    )
    host: str = Field(
        default="localhost", description="Redis server host", alias="REDIS_HOST"
    )
    port: int = Field(default=6379, description="Redis server port", alias="REDIS_PORT")
    db: int = Field(default=0, description="Redis database number", alias="REDIS_DB")
    password: Optional[str] = Field(
        default=None, description="Redis password", alias="REDIS_PASSWORD"
    )
    decode_responses: bool = Field(
        default=True, description="Decode Redis responses to strings"
    )
    socket_timeout: float = Field(default=30.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(
        default=30.0, description="Socket connect timeout in seconds"
    )


CacheSettings = Union[SqliteCacheSettings, RedisCacheSettings]

DEFAULT_SQLITE_SETTINGS = SqliteCacheSettings()
DEFAULT_REDIS_SETTINGS = RedisCacheSettings()


class BaseCacheProvider(ABC):
    """Abstract base class for cache providers."""

    @abstractmethod
    def hash_params(self, params: dict) -> str:
        """Generate hash for cache key from parameters."""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get cached response by key."""
        pass

    @abstractmethod
    def insert(self, key: str, request: dict, response: dict) -> None:
        """Insert new cache entry."""
        pass


class Sqlite3CacheProvider(BaseCacheProvider):
    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cache(
        key string PRIMARY KEY NOT NULL,
        request_params json NOT NULL,
        response json NOT NULL
    );
    """

    def __init__(self, settings: SqliteCacheSettings = DEFAULT_SQLITE_SETTINGS):
        self.db_path = settings.db_path
        logger.info(f"Using SQLite3 cache at {self.db_path}")

        # Ensure the directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database with proper settings
        self._init_database()

        # Thread-local storage for connections
        self._local = threading.local()

    def _init_database(self) -> None:
        """Initialize database with proper settings for concurrent access."""
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout to handle locks
            conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Create table
            conn.execute(self.CREATE_TABLE)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling and timeouts."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # 30 second timeout
                isolation_level=None,  # Autocommit mode
                check_same_thread=False,
            )
            # Configure connection for better concurrent access
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warning(f"Database locked, retrying after brief delay: {e}")
                time.sleep(0.1)  # Brief delay before retry
                # Retry once
                try:
                    if conn:
                        conn.close()
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=30.0,
                        isolation_level=None,
                        check_same_thread=False,
                    )
                    conn.execute("PRAGMA busy_timeout=30000")
                    conn.execute("PRAGMA journal_mode=WAL")
                    yield conn
                except Exception as retry_error:
                    logger.error(f"Database retry failed: {retry_error}")
                    raise
            else:
                logger.error(f"Database error: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as close_error:
                    logger.warning(f"Error closing database connection: {close_error}")

    def hash_params(self, params: dict) -> str:
        """Generate hash for cache key from parameters."""
        stringified = json.dumps(params, sort_keys=True).encode("utf-8")
        hashed = hashlib.md5(stringified).hexdigest()
        return hashed

    def get(self, key: str) -> Optional[str]:
        """Get cached response by key."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                res = cursor.execute(
                    "SELECT response FROM cache WHERE key = ?", (key,)
                ).fetchone()
                return res[0] if res else None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    def insert(self, key: str, request: dict, response: dict) -> None:
        """Insert new cache entry."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, request_params, response) VALUES (?, ?, ?)",
                    (
                        key,
                        json.dumps(request, sort_keys=True),
                        json.dumps(response, sort_keys=True),
                    ),
                )
                # No need to commit in autocommit mode
        except Exception as e:
            logger.warning(f"Cache insert failed for key {key}: {e}")
            # Don't raise exception for cache failures


class RedisCacheProvider(BaseCacheProvider):
    """Redis-based cache provider."""

    def __init__(self, settings: RedisCacheSettings = DEFAULT_REDIS_SETTINGS):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not available. Please install redis package: pip install redis"
            )

        self.settings = settings
        logger.info(f"Using Redis cache at {settings.host}:{settings.port}")

        # Initialize Redis connection
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection with proper settings."""
        try:
            connection_kwargs = {
                "host": self.settings.host,
                "port": self.settings.port,
                "db": self.settings.db,
                "decode_responses": self.settings.decode_responses,
                "socket_timeout": self.settings.socket_timeout,
                "socket_connect_timeout": self.settings.socket_connect_timeout,
            }

            # Only add password if it's not None
            if self.settings.password is not None:
                connection_kwargs["password"] = self.settings.password

            self.redis_client = redis.Redis(**connection_kwargs)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def hash_params(self, params: dict) -> str:
        """Generate hash for cache key from parameters."""
        stringified = json.dumps(params, sort_keys=True).encode("utf-8")
        hashed = hashlib.md5(stringified).hexdigest()
        return hashed

    def get(self, key: str) -> Optional[str]:
        """Get cached response by key."""
        try:
            result = self.redis_client.get(key)
            if result:
                # Parse the stored JSON to get the response
                cached_data = json.loads(result)
                return cached_data.get("response")
            return None
        except Exception as e:
            logger.warning(f"Redis cache get failed for key {key}: {e}")
            return None

    def insert(self, key: str, request: dict, response: dict) -> None:
        """Insert new cache entry."""
        try:
            # Store both request and response for consistency with SQLite
            cache_data = {"request_params": request, "response": response}
            self.redis_client.set(
                key,
                json.dumps(cache_data, sort_keys=True),
                # Optional: Set expiration time (e.g., 24 hours)
                # ex=86400
            )
        except Exception as e:
            logger.warning(f"Redis cache insert failed for key {key}: {e}")
            # Don't raise exception for cache failures


def get_llm_cache_provider(
    provider: Literal["sqlite", "redis"] = "sqlite",
    settings: Optional[CacheSettings] = None,
) -> BaseCacheProvider:
    """
    Get the LLM cache provider instance.

    Args:
        type: Type of cache provider ("sqlite" or "redis").
        settings: Settings for the cache provider. If None, uses default settings.

    Returns:
        BaseCacheProvider: An instance of the cache provider.
    """
    if provider == "sqlite":
        if settings is None:
            settings = DEFAULT_SQLITE_SETTINGS
        assert isinstance(
            settings, SqliteCacheSettings
        ), "For SQLite cache provider, settings must be an instance of SqliteCacheSettings"
        return Sqlite3CacheProvider(settings=settings)
    elif provider == "redis":
        if settings is None:
            settings = DEFAULT_REDIS_SETTINGS
        assert (
            type(settings) is RedisCacheSettings
        ), "For Redis cache provider, settings must be an instance of RedisCacheSettings"
        return RedisCacheProvider(settings=settings)
    else:
        raise ValueError(f"Unsupported cache provider type: {provider}")
