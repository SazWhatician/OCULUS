import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.config import settings
from app.telemetry import log_structured, metrics

logger = logging.getLogger("oculus.redis_bus")

UI_QUEUE_MAXSIZE = 256
DLQ_MAX_LEN = 2_000
IN_MEMORY_IDEMPOTENCY_MAX = 10_000
DEFAULT_RETRY_IDLE_MS = 1_500


class RedisBus:
    """
    Redis Streams event bus and shared-connection Pub/Sub manager:
      - Cross-container Redis Pub/Sub fan-out with single shared listener connection
      - Clean subscribe/unsubscribe lifecycle + bounded drop-oldest UI queues (zero OOM leaks)
      - True PEL retry tracking + XAUTOCLAIM recovery before routing to stream:dlq
      - Offline in-memory fallback with TTL idempotency, bounded streams, and retry parity
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Any = None
        self._pubsub: Any = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._pubsub_lock = asyncio.Lock()
        self._is_connected = False

        self._mock_streams: Dict[str, List[Dict[str, Any]]] = {}
        # Active local WebSocket queues keyed by channel (aliased as _mock_subscribers for test compat)
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._mock_subscribers = self._subscribers
        self._idempotency_map: "OrderedDict[str, float]" = OrderedDict()

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            # Keep connect timeout tight, but leave read socket_timeout=None so BLOCK/PubSub never spuriously timeout
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=None,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            await asyncio.wait_for(self._client.ping(), timeout=2.5)
            self._is_connected = True
            logger.info("Connected to Redis at %s", self.redis_url)

            # Attach any channels that subscribed before connect() completed
            async with self._pubsub_lock:
                if self._subscribers:
                    await self._ensure_pubsub_locked()
                    for channel in list(self._subscribers.keys()):
                        await self._pubsub.subscribe(channel)
        except Exception as exc:
            logger.warning("Redis not available at %s. Using in-memory fallback: %s", self.redis_url, exc)
            self._is_connected = False
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception:
                    pass
            self._client = None

    async def close(self) -> None:
        self._is_connected = False

        if self._pubsub_task is not None:
            self._pubsub_task.cancel()
            await asyncio.gather(self._pubsub_task, return_exceptions=True)
            self._pubsub_task = None

        if self._pubsub is not None:
            try:
                await self._pubsub.aclose()
            except AttributeError:
                await self._pubsub.close()
            except Exception as exc:
                logger.debug("Pub/Sub close warning: %s", exc)
            self._pubsub = None

        if self._client is not None:
            try:
                await self._client.aclose()
            except AttributeError:
                await self._client.close()
            except Exception as exc:
                logger.debug("Redis client close warning: %s", exc)
            self._client = None

        self._subscribers.clear()

    async def check_and_set_idempotency(self, key: str, ttl_seconds: int = 86_400) -> bool:
        """Returns True if key was newly set (not duplicate); False if already seen."""
        if self._is_connected and self._client:
            try:
                res = await self._client.set(key, "1", nx=True, ex=ttl_seconds)
                return bool(res)
            except Exception as exc:
                logger.error("Redis idempotency error: %s", exc)

        now = time.monotonic()
        existing_exp = self._idempotency_map.get(key)
        if existing_exp is not None and existing_exp > now:
            return False

        self._idempotency_map[key] = now + max(1, ttl_seconds)
        self._idempotency_map.move_to_end(key)
        while len(self._idempotency_map) > IN_MEMORY_IDEMPOTENCY_MAX:
            self._idempotency_map.popitem(last=False)
        return True

    async def publish_stream(self, stream: str, model: BaseModel, max_len: int = 5_000) -> str:
        """Publishes a Pydantic model to a Redis Stream with approximate trimming."""
        payload = model.model_dump_json()
        if self._is_connected and self._client:
            try:
                return await self._client.xadd(
                    stream,
                    {"payload": payload},
                    maxlen=max_len,
                    approximate=True,
                )
            except Exception as exc:
                logger.error("Error publishing to Redis stream %s: %s", stream, exc)
                raise

        bucket = self._mock_streams.setdefault(stream, [])
        msg_id = f"mock_{int(time.time() * 1000)}_{len(bucket) + 1}"
        bucket.append({"id": msg_id, "payload": payload, "attempts": 0, "next_retry_at": 0.0})
        if len(bucket) > max_len:
            del bucket[: len(bucket) - max_len]
        return msg_id

    async def ensure_group(self, stream: str, group: str) -> None:
        if self._is_connected and self._client:
            try:
                await self._client.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception as exc:
                if "BUSYGROUP" not in str(exc):
                    logger.debug("xgroup_create notice on %s/%s: %s", stream, group, exc)

    async def consume_stream(
        self,
        stream: str,
        group: str,
        consumer: str,
        model_cls: Type[BaseModel],
        handler: Callable[[Any], Awaitable[None]],
        stop_event: Optional[asyncio.Event] = None,
        block_ms: int = 2_000,
        dlq_max_retries: int = 3,
        retry_idle_ms: int = DEFAULT_RETRY_IDLE_MS,
    ) -> None:
        """
        Consumes from a Redis Stream with XREADGROUP, PEL retry tracking via XAUTOCLAIM,
        bounded retries (`dlq_max_retries`), dead-letter routing to `stream:dlq`, and XACK.
        """
        await self.ensure_group(stream, group)
        max_attempts = max(1, int(dlq_max_retries))
        reclaim_interval_sec = max(retry_idle_ms / 1000.0, 0.5)
        last_reclaim_mono = 0.0
        supports_autoclaim = True
        local_retries: Dict[str, int] = {}

        while not (stop_event and stop_event.is_set()):
            try:
                if self._is_connected and self._client:
                    batch: List[Tuple[str, Dict[str, Any]]] = []

                    # 1. Reclaim pending failed/orphaned PEL entries once retry_idle_ms has elapsed
                    now_mono = time.monotonic()
                    if (now_mono - last_reclaim_mono) >= reclaim_interval_sec:
                        last_reclaim_mono = now_mono
                        reclaimed, supports_autoclaim = await self._reclaim_pending(
                            stream=stream,
                            group=group,
                            consumer=consumer,
                            min_idle_ms=retry_idle_ms,
                            supports_autoclaim=supports_autoclaim,
                        )
                        batch.extend(reclaimed)

                    # 2. Read fresh entries from stream (non-blocking if reclaimed batch already queued)
                    read_block = 10 if batch else block_ms
                    entries = await self._client.xreadgroup(
                        groupname=group,
                        consumername=consumer,
                        streams={stream: ">"},
                        count=5,
                        block=read_block,
                    )
                    if entries:
                        for _, messages in entries:
                            batch.extend(messages)

                    if not batch:
                        continue

                    for msg_id, data in batch:
                        raw_payload = data.get("payload", "{}")
                        try:
                            event = model_cls.model_validate_json(raw_payload)
                            await handler(event)
                            await self._client.xack(stream, group, msg_id)
                            local_retries.pop(msg_id, None)
                        except Exception as err:
                            attempts = await self._get_delivery_attempts(
                                stream, group, msg_id, local_retries
                            )
                            if attempts >= max_attempts:
                                logger.error(
                                    "Message %s from %s failed %d/%d attempts; routing to stream:dlq: %s",
                                    msg_id,
                                    stream,
                                    attempts,
                                    max_attempts,
                                    err,
                                    exc_info=True,
                                )
                                await self._client.xadd(
                                    "stream:dlq",
                                    {
                                        "stream": stream,
                                        "group": group,
                                        "msg_id": msg_id,
                                        "attempts": str(attempts),
                                        "error": str(err)[:500],
                                        "payload": raw_payload,
                                    },
                                    maxlen=DLQ_MAX_LEN,
                                    approximate=True,
                                )
                                await self._client.xack(stream, group, msg_id)
                                local_retries.pop(msg_id, None)
                                metrics.inc("stream_dlq_routed")
                            else:
                                # Leave unacked in PEL so XAUTOCLAIM retries after retry_idle_ms
                                logger.warning(
                                    "Message %s from %s failed attempt %d/%d (%s); leaving in PEL for retry.",
                                    msg_id,
                                    stream,
                                    attempts,
                                    max_attempts,
                                    err,
                                )
                                metrics.inc("stream_retries")
                else:
                    # In-memory fallback with identical retry & DLQ semantics
                    processed_any = await self._consume_mock_step(
                        stream=stream,
                        group=group,
                        model_cls=model_cls,
                        handler=handler,
                        max_attempts=max_attempts,
                    )
                    if not processed_any:
                        await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in consumer loop for %s: %s", stream, exc)
                await asyncio.sleep(1.0)

    async def broadcast_ui(self, session_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publishes JSON envelope to Redis Pub/Sub `channel:session:{session_id}`.
        When Redis is active, local subscribers receive the event via the Pub/Sub listener
        (guaranteeing cross-container delivery with zero local duplication).
        """
        channel = f"channel:session:{session_id}"
        envelope = json.dumps(
            {"event_type": event_type, "session_id": session_id, "data": data},
            default=str,
        )

        if self._is_connected and self._client:
            try:
                await self._client.publish(channel, envelope)
                return
            except Exception as exc:
                logger.error("Failed publishing to Redis %s (%s); falling back locally.", channel, exc)

        self._enqueue_local(channel, envelope)

    async def subscribe_ui(self, session_id: str) -> asyncio.Queue:
        """
        Registers a bounded asyncio.Queue for `channel:session:{session_id}` and opens
        a Redis Pub/Sub channel subscription on the shared listener connection.
        """
        channel = f"channel:session:{session_id}"
        q: asyncio.Queue = asyncio.Queue(maxsize=UI_QUEUE_MAXSIZE)

        async with self._pubsub_lock:
            is_first = channel not in self._subscribers
            self._subscribers.setdefault(channel, []).append(q)

            if self._is_connected and self._client:
                await self._ensure_pubsub_locked()
                if is_first:
                    await self._pubsub.subscribe(channel)

        return q

    async def unsubscribe_ui(self, session_id: str, queue: asyncio.Queue) -> None:
        """Removes a closed WebSocket queue and unsubscribes from Redis when no listeners remain."""
        channel = f"channel:session:{session_id}"

        async with self._pubsub_lock:
            queues = self._subscribers.get(channel)
            if not queues:
                return
            try:
                queues.remove(queue)
            except ValueError:
                pass

            if not queues:
                self._subscribers.pop(channel, None)
                if self._pubsub is not None:
                    try:
                        await self._pubsub.unsubscribe(channel)
                    except Exception as exc:
                        logger.debug("Redis unsubscribe warning for %s: %s", channel, exc)

    # -----------------------------------------------------------------------
    # Internal Pub/Sub & Stream Recovery Helpers
    # -----------------------------------------------------------------------
    async def _ensure_pubsub_locked(self) -> None:
        if self._pubsub is None and self._client is not None:
            self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        if self._pubsub_task is None or self._pubsub_task.done():
            self._pubsub_task = asyncio.create_task(self._pubsub_listener_loop())

    async def _pubsub_listener_loop(self) -> None:
        """Single shared Redis Pub/Sub reader fanning out messages to all local WS queues."""
        while self._is_connected:
            try:
                if self._pubsub is None or not self._subscribers:
                    await asyncio.sleep(0.1)
                    continue

                msg = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg or msg.get("type") != "message":
                    continue

                channel = msg.get("channel")
                raw_data = msg.get("data")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8", errors="replace")

                if isinstance(channel, str) and isinstance(raw_data, str):
                    self._enqueue_local(channel, raw_data)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Redis Pub/Sub listener error (%s); attempting resubscribe.", exc)
                await asyncio.sleep(1.0)
                await self._resubscribe_all()

    async def _resubscribe_all(self) -> None:
        if not (self._is_connected and self._client):
            return
        async with self._pubsub_lock:
            if self._pubsub is not None:
                try:
                    await self._pubsub.aclose()
                except Exception:
                    pass
                self._pubsub = None
            if self._subscribers:
                self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
                try:
                    await self._pubsub.subscribe(*list(self._subscribers.keys()))
                    logger.info("Resubscribed %d UI Pub/Sub channel(s).", len(self._subscribers))
                except Exception as exc:
                    logger.error("Failed resubscribing Pub/Sub channels: %s", exc)

    def _enqueue_local(self, channel: str, envelope: str) -> None:
        queues = self._subscribers.get(channel)
        if not queues:
            return
        for q in list(queues):
            if q.full():
                try:
                    q.get_nowait()  # Drop oldest telemetry frame so slow clients never stall fan-out
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass

    async def _reclaim_pending(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int,
        supports_autoclaim: bool,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], bool]:
        if supports_autoclaim:
            try:
                res = await self._client.xautoclaim(
                    name=stream,
                    groupname=group,
                    consumername=consumer,
                    min_idle_time=min_idle_ms,
                    start_id="0-0",
                    count=5,
                )
                # redis-py returns (next_id, messages, deleted_ids) or (next_id, messages)
                messages = res[1] if isinstance(res, (tuple, list)) and len(res) >= 2 else []
                return ([(mid, fields) for mid, fields in messages if fields], True)
            except Exception as exc:
                if "unknown command" in str(exc).lower():
                    supports_autoclaim = False
                else:
                    logger.debug("xautoclaim notice on %s: %s", stream, exc)
                    return ([], True)

        # Fallback for Redis < 6.2 using XPENDING + XCLAIM
        try:
            pending = await self._client.xpending_range(
                name=stream,
                groupname=group,
                min="-",
                max="+",
                count=5,
            )
            stale_ids = [
                p["message_id"]
                for p in pending
                if int(p.get("time_since_delivered", 0)) >= min_idle_ms
            ]
            if not stale_ids:
                return ([], False)
            claimed = await self._client.xclaim(
                name=stream,
                groupname=group,
                consumername=consumer,
                min_idle_time=min_idle_ms,
                message_ids=stale_ids,
            )
            return ([(mid, fields) for mid, fields in claimed if fields], False)
        except Exception as exc:
            logger.debug("xclaim fallback notice on %s: %s", stream, exc)
            return ([], False)

    async def _get_delivery_attempts(
        self,
        stream: str,
        group: str,
        msg_id: str,
        local_retries: Dict[str, int],
    ) -> int:
        redis_attempts = 0
        try:
            info = await self._client.xpending_range(
                name=stream,
                groupname=group,
                min=msg_id,
                max=msg_id,
                count=1,
            )
            if info:
                redis_attempts = int(info[0].get("times_delivered", 1))
        except Exception:
            pass

        attempts = max(redis_attempts, local_retries.get(msg_id, 0) + 1)
        local_retries[msg_id] = attempts
        return attempts

    async def _consume_mock_step(
        self,
        stream: str,
        group: str,
        model_cls: Type[BaseModel],
        handler: Callable[[Any], Awaitable[None]],
        max_attempts: int,
    ) -> bool:
        bucket = self._mock_streams.get(stream)
        if not bucket:
            return False

        now = time.monotonic()
        ready_idx = next(
            (i for i, item in enumerate(bucket) if float(item.get("next_retry_at", 0.0)) <= now),
            None,
        )
        if ready_idx is None:
            return False

        item = bucket.pop(ready_idx)
        msg_id = item.get("id", "mock")
        raw_payload = item.get("payload", "{}")

        try:
            event = model_cls.model_validate_json(raw_payload)
            await handler(event)
        except Exception as err:
            attempts = int(item.get("attempts", 0)) + 1
            if attempts >= max_attempts:
                logger.error(
                    "Mock message %s from %s failed %d/%d attempts; routing to stream:dlq: %s",
                    msg_id,
                    stream,
                    attempts,
                    max_attempts,
                    err,
                )
                dlq = self._mock_streams.setdefault("stream:dlq", [])
                dlq.append(
                    {
                        "id": f"dlq_{msg_id}",
                        "stream": stream,
                        "group": group,
                        "msg_id": msg_id,
                        "attempts": attempts,
                        "error": str(err)[:500],
                        "payload": raw_payload,
                    }
                )
                if len(dlq) > DLQ_MAX_LEN:
                    del dlq[: len(dlq) - DLQ_MAX_LEN]
                metrics.inc("stream_dlq_routed")
            else:
                delay = min(0.05 * (2 ** (attempts - 1)), 0.5)
                item["attempts"] = attempts
                item["next_retry_at"] = time.monotonic() + delay
                bucket.append(item)
                metrics.inc("stream_retries")
        return True


redis_bus = RedisBus()