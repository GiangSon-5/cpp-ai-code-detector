"""
shared/message_broker.py — Redpanda / Kafka Producer & Consumer helpers.

Wraps confluent-kafka with JSON serialisation via orjson and
automatic Deep Logging.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

import orjson
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()


# =====================================================================
# Topics
# =====================================================================
class Topics:
    CODE_SUBMITTED = "code.submitted"
    PREDICTION_COMPLETED = "prediction.completed"
    RETRAIN_TRIGGER = "retrain.trigger"

    ALL = [CODE_SUBMITTED, PREDICTION_COMPLETED, RETRAIN_TRIGGER]


# =====================================================================
# Producer
# =====================================================================
class EventProducer:
    """Singleton Redpanda/Kafka producer."""

    _instance: Optional["EventProducer"] = None

    def __new__(cls) -> "EventProducer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_ready"):
            return
        self._ready = True
        self._producer = Producer({
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "client.id": "cpp-detector-producer",
            "acks": "all",
        })

    @AppLogger.log_function(module="message_broker")
    def publish(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        """Publish a JSON event to a Redpanda topic."""
        raw = orjson.dumps(value)
        self._producer.produce(
            topic=topic,
            value=raw,
            key=key.encode("utf-8") if key else None,
            callback=self._delivery_report,
        )
        self._producer.flush(timeout=5)

    @staticmethod
    def _delivery_report(err: Any, msg: Any) -> None:
        if err is not None:
            logger.error(
                module="message_broker",
                function="_delivery_report",
                message=f"Delivery failed: {err}",
                error=str(err),
            )


# =====================================================================
# Consumer
# =====================================================================
class EventConsumer:
    """Reusable Redpanda/Kafka consumer."""

    def __init__(self, group_id: str, topics: list[str]) -> None:
        self._consumer = Consumer({
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self._consumer.subscribe(topics)
        self._running = False

    def consume_loop(self, handler: Callable[[str, dict[str, Any]], None], poll_timeout: float = 1.0) -> None:
        """Blocking consume loop.  Calls handler(topic, payload_dict) for each message."""
        self._running = True
        logger.info(
            module="message_broker",
            function="consume_loop",
            message="Consumer loop started",
            input_data={"topics": self._consumer.subscription()},
        )
        try:
            while self._running:
                msg = self._consumer.poll(timeout=poll_timeout)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(
                        module="message_broker",
                        function="consume_loop",
                        error=str(msg.error()),
                    )
                    continue

                t0 = time.perf_counter()
                try:
                    payload = orjson.loads(msg.value())
                    handler(msg.topic(), payload)
                    latency = (time.perf_counter() - t0) * 1000
                    logger.info(
                        module="message_broker",
                        function="consume_loop",
                        message=f"Processed message from {msg.topic()}",
                        input_data=payload,
                        latency_ms=latency,
                    )
                except Exception as exc:
                    latency = (time.perf_counter() - t0) * 1000
                    logger.error(
                        module="message_broker",
                        function="consume_loop",
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=latency,
                    )
        finally:
            self._consumer.close()

    def stop(self) -> None:
        self._running = False


# =====================================================================
# Topic creation helper
# =====================================================================
@AppLogger.log_function(module="message_broker")
def ensure_topics_exist(num_partitions: int = 1, replication_factor: int = 1) -> dict[str, Any]:
    """Create all required topics if they don't already exist."""
    admin = AdminClient({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
    existing = admin.list_topics(timeout=10).topics.keys()

    to_create = [
        NewTopic(t, num_partitions=num_partitions, replication_factor=replication_factor)
        for t in Topics.ALL
        if t not in existing
    ]
    if not to_create:
        return {"status": "all_topics_exist", "topics": Topics.ALL}

    futures = admin.create_topics(to_create)
    results = {}
    for topic, future in futures.items():
        try:
            future.result()
            results[topic] = "created"
        except KafkaException as exc:
            results[topic] = f"error: {exc}"
    return results
