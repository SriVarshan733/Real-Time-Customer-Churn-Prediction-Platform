"""
Kafka Producer — Customer Event Simulator
Simulates a real-world event stream of customer behaviour data
sent from a CRM / product analytics system.
"""

import json
import random
import time
import uuid
from datetime import datetime, timedelta

from confluent_kafka import Producer
from loguru import logger

# ─── Config ───────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "customer-events"
EVENTS_PER_SECOND = 100

# ─── Data Generators ──────────────────────────────────────────────────────────

PLANS = ["Basic", "Standard", "Premium", "Enterprise"]
PAYMENT_METHODS = ["credit_card", "bank_transfer", "paypal", "direct_debit"]
CHURN_RISK_SEGMENTS = ["low", "medium", "high", "critical"]


def generate_customer_event() -> dict:
    """
    Generates a realistic customer event record.
    In production this would come from your CRM, Segment, Mixpanel, etc.
    """
    customer_id = f"C_{random.randint(10000, 99999)}"
    tenure_months = random.randint(1, 72)

    # Correlated features — churners tend to have more support calls,
    # lower engagement, and recent billing issues
    is_at_risk = random.random() < 0.3

    return {
        "event_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "event_type": "customer_snapshot",
        "timestamp": datetime.utcnow().isoformat(),

        # Demographic features
        "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
        "region": random.choice(["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]),

        # Contract features
        "plan": random.choice(PLANS),
        "contract_type": random.choice(["monthly", "annual", "biennial"]),
        "payment_method": random.choice(PAYMENT_METHODS),
        "tenure_months": tenure_months,
        "monthly_charges": round(random.uniform(20, 150), 2),
        "total_charges": round(random.uniform(50, 8000), 2),

        # Usage features
        "avg_monthly_usage_gb": round(random.uniform(1, 500), 2),
        "num_logins_last_30d": random.randint(0, 120) if not is_at_risk else random.randint(0, 20),
        "num_features_used": random.randint(1, 20),
        "last_login_days_ago": random.randint(0, 10) if not is_at_risk else random.randint(15, 90),

        # Support features
        "num_support_calls_3m": random.randint(0, 2) if not is_at_risk else random.randint(3, 15),
        "num_complaints_6m": random.randint(0, 1) if not is_at_risk else random.randint(2, 8),
        "avg_support_resolution_hours": round(random.uniform(1, 72), 1),

        # Billing features
        "num_late_payments": random.randint(0, 1) if not is_at_risk else random.randint(2, 10),
        "billing_issues_flag": 1 if is_at_risk and random.random() > 0.5 else 0,

        # Engagement features
        "nps_score": random.randint(7, 10) if not is_at_risk else random.randint(0, 6),
        "email_open_rate": round(random.uniform(0.3, 0.9), 3) if not is_at_risk else round(random.uniform(0, 0.3), 3),
        "referrals_made": random.randint(0, 5),

        # Label (available only in historical data for training)
        "churned": 1 if is_at_risk and random.random() > 0.4 else 0,
    }


def delivery_report(err, msg):
    """Callback invoked on message delivery success or failure."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")


# ─── Producer ─────────────────────────────────────────────────────────────────

def run_producer(n_events: int = None, events_per_second: int = EVENTS_PER_SECOND):
    """
    Streams customer events to Kafka.
    Args:
        n_events: Number of events to produce. None = run indefinitely.
        events_per_second: Throughput rate.
    """
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "client.id": "churn-producer",
        "acks": "all",                  # Wait for all replicas
        "retries": 3,
        "batch.size": 16384,
        "linger.ms": 5,                 # Small batching window for throughput
        "compression.type": "snappy",
    }

    producer = Producer(conf)
    logger.info(f"Producer started → topic: {TOPIC} @ {events_per_second} events/sec")

    count = 0
    sleep_time = 1.0 / events_per_second

    try:
        while n_events is None or count < n_events:
            event = generate_customer_event()
            key = event["customer_id"]
            value = json.dumps(event)

            producer.produce(
                topic=TOPIC,
                key=key.encode("utf-8"),
                value=value.encode("utf-8"),
                callback=delivery_report,
            )

            # Poll to handle delivery callbacks
            producer.poll(0)

            count += 1
            if count % 1000 == 0:
                logger.info(f"Produced {count:,} events")

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Producer stopped by user")
    finally:
        # Flush remaining messages
        producer.flush()
        logger.info(f"Total events produced: {count:,}")


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def main(
        n_events: int = typer.Option(None, help="Number of events (None = infinite)"),
        rate: int = typer.Option(100, help="Events per second"),
    ):
        run_producer(n_events=n_events, events_per_second=rate)

    app()