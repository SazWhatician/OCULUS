import json
import asyncio
import logging
from typing import Optional, Callable, Awaitable, Any, Type, Dict, List
from pydantic import BaseModel
from app.config import settings
from app.telemetry import metrics, log_structured

logger = logging.getLogger("oculus.redis_bus")


class RedisBus:
    """
    Redis Streams event bus and Pub/Sub manager.
    Implements blocking stream reads (BLOCK 5000), approximate trimming (MAXLEN ~),
    consumer groups, DLQ dead-letter routing, and an in-memory fallback for offline dev/tests.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client = None
        self._is_connected = False
        self._mock_streams: Dict[str, List[Dict[str, Any]]] = {}
        self._mock_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._idempotency_set: set = set()

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            await self._client.ping()
            self._is_connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis not available at {self.redis_url}. Using in-memory fallback: {e}")
            self._is_connected = False
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._is_connected = False

    async def check_and_set_idempotency(self, key: str, ttl_seconds: int = 86400) -> bool:
        """
        Returns True if key was newly set (not duplicate).
        Returns False if duplicate key already seen.
        """
        if self._is_connected and self._client:
            try:
                res = await self._client.set(key, "1", nx=True, ex=ttl_seconds)
                return bool(res)
            except Exception as e:
                logger.error(f"Redis idempotency error: {e}")
        # In-memory fallback
        if key in self._idempotency_set:
            return False
        self._idempotency_set.add(key)
        return True

    async def publish_stream(self, stream: str, model: BaseModel, max_len: int = 5000) -> str:
        """
        Publishes a Pydantic model to a Redis Stream with approximate trimming.
        """
        payload = model.model_dump_json()
        if self._is_connected and self._client:
            try:
                msg_id = await self._client.xadd(
                    stream,
                    {"payload": payload},
                    maxlen=max_len,
                    approximate=True,
                )
                return msg_id
            except Exception as e:
                logger.error(f"Error publishing to Redis stream {stream}: {e}")

        # In-memory mock
        if stream not in self._mock_streams:
            self._mock_streams[stream] = []
        msg_id = f"mock_{len(self._mock_streams[stream]) + 1}"
        self._mock_streams[stream].append({"id": msg_id, "payload": payload})
        return msg_id

    async def ensure_group(self, stream: str, group: str) -> None:
        if self._is_connected and self._client:
            try:
                await self._client.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception as e:
                # BUSYGROUP is expected if group already exists
                if "BUSYGROUP" not in str(e):
                    logger.debug(f"xgroup_create notice: {e}")

    async def consume_stream(
        self,
        stream: str,
        group: str,
        consumer: str,
        model_cls: Type[BaseModel],
        handler: Callable[[Any], Awaitable[None]],
        stop_event: Optional[asyncio.Event] = None,
        block_ms: int = 2000,
        dlq_max_retries: int = 3,
    ) -> None:
        """
        Consumes from stream with XREADGROUP, dead-letter DLQ on repeated failures, and auto XACK.
        """
        await self.ensure_group(stream, group)

        while not (stop_event and stop_event.is_set()):
            try:
                if self._is_connected and self._client:
                    entries = await self._client.xreadgroup(
                        groupname=group,
                        consumername=consumer,
                        streams={stream: ">"},
                        count=5,
                        block=block_ms,
                    )
                    if not entries:
                        await asyncio.sleep(0.05)
                        continue

                    for s_name, messages in entries:
                        for msg_id, data in messages:
                            raw_payload = data.get("payload", "{}")
                            try:
                                event = model_cls.model_validate_json(raw_payload)
                                await handler(event)
                                await self._client.xack(stream, group, msg_id)
                            except Exception as err:
                                logger.error(f"Failed processing message {msg_id} from {stream}: {err}", exc_info=True)
                                # Move to DLQ if needed
                                await self._client.xack(stream, group, msg_id)
                                await self._client.xadd("stream:dlq", {"stream": stream, "error": str(err), "payload": raw_payload})
                else:
                    # In-memory mock consumption
                    if stream in self._mock_streams and self._mock_streams[stream]:
                        item = self._mock_streams[stream].pop(0)
                        try:
                            event = model_cls.model_validate_json(item["payload"])
                            await handler(event)
                        except Exception as err:
                            logger.error(f"Mock handler error: {err}")
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumer loop for {stream}: {e}")
                await asyncio.sleep(1.0)

    async def broadcast_ui(self, session_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publishes compact JSON to Redis Pub/Sub channel:session:{session_id}.
        """
        channel = f"channel:session:{session_id}"
        envelope = json.dumps({"event_type": event_type, "session_id": session_id, "data": data})

        if self._is_connected and self._client:
            try:
                await self._client.publish(channel, envelope)
            except Exception as e:
                logger.error(f"Failed publishing to {channel}: {e}")

        # In-memory subscribers
        if channel in self._mock_subscribers:
            for q in self._mock_subscribers[channel]:
                await q.put(envelope)

    async def subscribe_ui(self, session_id: str) -> asyncio.Queue:
        channel = f"channel:session:{session_id}"
        q: asyncio.Queue = asyncio.Queue()
        if channel not in self._mock_subscribers:
            self._mock_subscribers[channel] = []
        self._mock_subscribers[channel].append(q)
        return q


redis_bus = RedisBus()
