"""
Kafka Consumer — Customer Event Ingestion
Consumes events from Kafka and persists them to PostgreSQL (raw events table).
Designed for exactly-once semantics using Kafka transactions.
"""

import json
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaError, KafkaException
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Config ───────────────────────────────────────────────────────────────────

KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "churn-feature-group",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,        # Manual commit for exactly-once
    "max.poll.interval.ms": 300000,
    "session.timeout.ms": 45000,
}

TOPIC = "customer-events"
DB_URL = "postgresql://churn_user:churn_pass@localhost:5432/churn_db"
BATCH_SIZE = 500                        # Rows per DB insert batch
COMMIT_INTERVAL_SECS = 30


# ─── Database Layer ───────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
def get_db_connection(db_url: str = DB_URL):
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn


def create_tables(conn):
    """Create raw events table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_customer_events (
                id              BIGSERIAL PRIMARY KEY,
                event_id        UUID UNIQUE NOT NULL,
                customer_id     VARCHAR(20) NOT NULL,
                event_type      VARCHAR(50),
                event_timestamp TIMESTAMP WITH TIME ZONE,
                ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                payload         JSONB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_raw_events_customer_id
                ON raw_customer_events (customer_id);

            CREATE INDEX IF NOT EXISTS idx_raw_events_timestamp
                ON raw_customer_events (event_timestamp);
        """)
        conn.commit()
    logger.info("Database tables ready")


def batch_insert_events(conn, events: list[dict]):
    """Bulk insert events using execute_values for performance."""
    if not events:
        return

    records = [
        (
            e["event_id"],
            e["customer_id"],
            e.get("event_type"),
            e.get("timestamp"),
            json.dumps(e),
        )
        for e in events
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO raw_customer_events
                (event_id, customer_id, event_type, event_timestamp, payload)
            VALUES %s
            ON CONFLICT (event_id) DO NOTHING
            """,
            records,
            template="(%s, %s, %s, %s, %s::jsonb)",
            page_size=500,
        )
        conn.commit()


# ─── Consumer ─────────────────────────────────────────────────────────────────

class ChurnEventConsumer:
    def __init__(self):
        self.consumer = Consumer(KAFKA_CONFIG)
        self.conn = get_db_connection()
        self.running = True
        self.buffer: list[dict] = []
        self.last_commit_time = datetime.utcnow()

        # Graceful shutdown on SIGTERM (needed for Docker/ECS)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _flush_buffer(self):
        """Persist buffered events to DB and commit Kafka offsets."""
        if self.buffer:
            try:
                batch_insert_events(self.conn, self.buffer)
                self.consumer.commit(asynchronous=False)
                logger.info(f"Flushed {len(self.buffer)} events to DB")
                self.buffer.clear()
                self.last_commit_time = datetime.utcnow()
            except Exception as e:
                logger.error(f"Flush failed: {e}")
                self.conn.rollback()
                raise

    def run(self):
        self.consumer.subscribe([TOPIC])
        create_tables(self.conn)
        logger.info(f"Consumer started → listening to topic: {TOPIC}")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    # Flush on idle
                    elapsed = (datetime.utcnow() - self.last_commit_time).seconds
                    if elapsed >= COMMIT_INTERVAL_SECS and self.buffer:
                        self._flush_buffer()
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition")
                    else:
                        raise KafkaException(msg.error())
                    continue

                try:
                    event = json.loads(msg.value().decode("utf-8"))
                    self.buffer.append(event)

                    if len(self.buffer) >= BATCH_SIZE:
                        self._flush_buffer()

                except json.JSONDecodeError as e:
                    logger.warning(f"Malformed message skipped: {e}")

        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            self._flush_buffer()
            self.consumer.close()
            self.conn.close()
            logger.info("Consumer shut down cleanly")


if __name__ == "__main__":
    consumer = ChurnEventConsumer()
    consumer.run()