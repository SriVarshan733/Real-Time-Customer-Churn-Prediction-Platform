"""
PySpark Feature Engineering Pipeline
Transforms raw customer events into a rich feature set for model training.
Handles: aggregations, time-window features, derived ratios, encoding.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType
)
from loguru import logger
import os

# ─── Spark Session ────────────────────────────────────────────────────────────

def create_spark_session(app_name: str = "ChurnFeatureEngineering") -> SparkSession:
    """
    Creates a Spark session configured for local dev and AWS EMR.
    On EMR, Spark picks up cluster config automatically.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.1"
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_raw_events(spark: SparkSession, jdbc_url: str, table: str = "raw_customer_events") -> DataFrame:
    """Load raw events from PostgreSQL via JDBC."""
    return (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", "churn_user")
        .option("password", "churn_pass")
        .option("driver", "org.postgresql.Driver")
        .option("numPartitions", "10")
        .option("partitionColumn", "id")
        .option("lowerBound", "1")
        .option("upperBound", "10000000")
        .load()
    )


# ─── Feature Engineering Steps ────────────────────────────────────────────────

def extract_payload_fields(df: DataFrame) -> DataFrame:
    """Flatten JSONB payload column into typed columns."""
    logger.info("Step 1: Extracting payload fields...")
    return df.select(
        F.col("customer_id"),
        F.col("event_timestamp"),
        F.col("ingested_at"),

        # Extract typed fields from JSONB payload
        F.get_json_object("payload", "$.age_group").cast(StringType()).alias("age_group"),
        F.get_json_object("payload", "$.region").cast(StringType()).alias("region"),
        F.get_json_object("payload", "$.plan").cast(StringType()).alias("plan"),
        F.get_json_object("payload", "$.contract_type").cast(StringType()).alias("contract_type"),
        F.get_json_object("payload", "$.payment_method").cast(StringType()).alias("payment_method"),
        F.get_json_object("payload", "$.tenure_months").cast(IntegerType()).alias("tenure_months"),
        F.get_json_object("payload", "$.monthly_charges").cast(DoubleType()).alias("monthly_charges"),
        F.get_json_object("payload", "$.total_charges").cast(DoubleType()).alias("total_charges"),
        F.get_json_object("payload", "$.avg_monthly_usage_gb").cast(DoubleType()).alias("avg_monthly_usage_gb"),
        F.get_json_object("payload", "$.num_logins_last_30d").cast(IntegerType()).alias("num_logins_last_30d"),
        F.get_json_object("payload", "$.num_features_used").cast(IntegerType()).alias("num_features_used"),
        F.get_json_object("payload", "$.last_login_days_ago").cast(IntegerType()).alias("last_login_days_ago"),
        F.get_json_object("payload", "$.num_support_calls_3m").cast(IntegerType()).alias("num_support_calls_3m"),
        F.get_json_object("payload", "$.num_complaints_6m").cast(IntegerType()).alias("num_complaints_6m"),
        F.get_json_object("payload", "$.avg_support_resolution_hours").cast(DoubleType()).alias("avg_support_resolution_hours"),
        F.get_json_object("payload", "$.num_late_payments").cast(IntegerType()).alias("num_late_payments"),
        F.get_json_object("payload", "$.billing_issues_flag").cast(IntegerType()).alias("billing_issues_flag"),
        F.get_json_object("payload", "$.nps_score").cast(IntegerType()).alias("nps_score"),
        F.get_json_object("payload", "$.email_open_rate").cast(DoubleType()).alias("email_open_rate"),
        F.get_json_object("payload", "$.referrals_made").cast(IntegerType()).alias("referrals_made"),
        F.get_json_object("payload", "$.churned").cast(IntegerType()).alias("churned"),
    )


def engineer_derived_features(df: DataFrame) -> DataFrame:
    """
    Create derived features that significantly boost model performance.
    These are the kind of features that separate good DS from great DS.
    """
    logger.info("Step 2: Engineering derived features...")
    return df.withColumns({
        # Revenue signals
        "charges_per_tenure": F.col("monthly_charges") / (F.col("tenure_months") + 1),
        "total_vs_expected_ratio": F.col("total_charges") / (
            F.col("monthly_charges") * F.col("tenure_months") + 1
        ),

        # Engagement signals
        "logins_per_feature": F.col("num_logins_last_30d") / (F.col("num_features_used") + 1),
        "is_dormant": (F.col("last_login_days_ago") > 30).cast(IntegerType()),
        "engagement_score": (
            F.col("num_logins_last_30d") * 0.4 +
            F.col("num_features_used") * 0.3 +
            F.col("email_open_rate") * 10 * 0.3
        ),

        # Support distress signals
        "support_distress_score": (
            F.col("num_support_calls_3m") * 2 +
            F.col("num_complaints_6m") * 3 +
            F.col("billing_issues_flag") * 2 +
            F.col("num_late_payments")
        ),
        "calls_per_tenure": F.col("num_support_calls_3m") / (F.col("tenure_months") + 1),

        # Satisfaction signals
        "is_promoter": (F.col("nps_score") >= 9).cast(IntegerType()),
        "is_detractor": (F.col("nps_score") <= 6).cast(IntegerType()),
        "nps_group": F.when(F.col("nps_score") >= 9, "promoter")
                      .when(F.col("nps_score") >= 7, "passive")
                      .otherwise("detractor"),

        # Loyalty signals
        "is_long_tenure": (F.col("tenure_months") >= 24).cast(IntegerType()),
        "tenure_bucket": F.when(F.col("tenure_months") < 6, "new")
                          .when(F.col("tenure_months") < 12, "growing")
                          .when(F.col("tenure_months") < 24, "established")
                          .otherwise("loyal"),

        # Composite churn risk score (used as a feature, not the label)
        "risk_composite": (
            F.col("num_support_calls_3m") * 0.25 +
            F.col("num_complaints_6m") * 0.20 +
            (1 - F.col("email_open_rate")) * 0.15 +
            (F.col("last_login_days_ago") / 90).cast(DoubleType()) * 0.20 +
            F.col("num_late_payments") * 0.20
        ),
    })


def encode_categoricals(df: DataFrame) -> DataFrame:
    """One-hot encode categorical features via Spark ML StringIndexer alternative."""
    logger.info("Step 3: Encoding categorical features...")

    # Plan encoding
    df = df.withColumn("plan_encoded",
        F.when(F.col("plan") == "Basic", 0)
         .when(F.col("plan") == "Standard", 1)
         .when(F.col("plan") == "Premium", 2)
         .when(F.col("plan") == "Enterprise", 3)
         .otherwise(0)
    )

    # Contract type encoding
    df = df.withColumn("contract_encoded",
        F.when(F.col("contract_type") == "monthly", 0)
         .when(F.col("contract_type") == "annual", 1)
         .when(F.col("contract_type") == "biennial", 2)
         .otherwise(0)
    )

    # Payment method encoding
    df = df.withColumn("payment_encoded",
        F.when(F.col("payment_method") == "credit_card", 0)
         .when(F.col("payment_method") == "bank_transfer", 1)
         .when(F.col("payment_method") == "paypal", 2)
         .when(F.col("payment_method") == "direct_debit", 3)
         .otherwise(0)
    )

    return df


def add_window_features(df: DataFrame) -> DataFrame:
    """
    Add rolling window aggregations per customer.
    Shows trajectory over time — crucial for churn detection.
    """
    logger.info("Step 4: Adding window/lag features...")

    window_customer = Window.partitionBy("customer_id").orderBy("event_timestamp")
    window_30d = Window.partitionBy("customer_id").orderBy(
        F.col("event_timestamp").cast("long")
    ).rangeBetween(-30 * 86400, 0)

    return df.withColumns({
        "rolling_avg_logins_30d": F.avg("num_logins_last_30d").over(window_30d),
        "rolling_avg_support_calls": F.avg("num_support_calls_3m").over(window_30d),
        "prev_monthly_charges": F.lag("monthly_charges", 1).over(window_customer),
        "charges_change_pct": (
            (F.col("monthly_charges") - F.lag("monthly_charges", 1).over(window_customer)) /
            (F.lag("monthly_charges", 1).over(window_customer) + 1)
        ),
    })


def remove_nulls_and_outliers(df: DataFrame) -> DataFrame:
    """Clean data: remove rows with critical nulls, cap outliers."""
    logger.info("Step 5: Cleaning data...")

    # Drop rows missing critical fields
    df = df.dropna(subset=["customer_id", "churned", "tenure_months", "monthly_charges"])

    # Cap outliers at 99th percentile using approxQuantile
    # (done in pandas post-collect for simplicity in this step)
    df = df.filter(
        (F.col("monthly_charges").between(5, 500)) &
        (F.col("tenure_months").between(0, 120)) &
        (F.col("num_support_calls_3m").between(0, 50))
    )

    # Fill derived nulls (from window functions on first row)
    df = df.fillna({
        "rolling_avg_logins_30d": 0.0,
        "rolling_avg_support_calls": 0.0,
        "prev_monthly_charges": 0.0,
        "charges_change_pct": 0.0,
    })

    return df


def select_final_feature_set(df: DataFrame) -> DataFrame:
    """Select the final 40+ features used for model training."""
    logger.info("Step 6: Selecting final feature set...")

    return df.select(
        "customer_id",
        "event_timestamp",
        # Numeric features
        "tenure_months", "monthly_charges", "total_charges",
        "avg_monthly_usage_gb", "num_logins_last_30d", "num_features_used",
        "last_login_days_ago", "num_support_calls_3m", "num_complaints_6m",
        "avg_support_resolution_hours", "num_late_payments", "nps_score",
        "email_open_rate", "referrals_made",
        # Encoded categoricals
        "plan_encoded", "contract_encoded", "payment_encoded",
        # Derived features
        "charges_per_tenure", "total_vs_expected_ratio",
        "logins_per_feature", "is_dormant", "engagement_score",
        "support_distress_score", "calls_per_tenure",
        "is_promoter", "is_detractor", "is_long_tenure",
        "risk_composite",
        # Window features
        "rolling_avg_logins_30d", "rolling_avg_support_calls",
        "prev_monthly_charges", "charges_change_pct",
        # Target
        "churned",
    )


# ─── Write to Feature Store ───────────────────────────────────────────────────

def write_to_feature_store(df: DataFrame, jdbc_url: str):
    """Write engineered features to PostgreSQL feature store."""
    logger.info("Step 7: Writing to feature store...")
    (
        df.write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "customer_features")
        .option("user", "churn_user")
        .option("password", "churn_pass")
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "10000")
        .mode("overwrite")
        .save()
    )
    logger.info(f"Feature store updated with {df.count():,} records")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    spark = create_spark_session()
    jdbc_url = "jdbc:postgresql://localhost:5432/churn_db"

    logger.info("=== Starting Feature Engineering Pipeline ===")

    df = load_raw_events(spark, jdbc_url)
    logger.info(f"Loaded {df.count():,} raw events")

    df = extract_payload_fields(df)
    df = engineer_derived_features(df)
    df = encode_categoricals(df)
    df = add_window_features(df)
    df = remove_nulls_and_outliers(df)
    df = select_final_feature_set(df)

    logger.info(f"Feature engineering complete. Final shape: ({df.count():,}, {len(df.columns)})")
    logger.info(f"Churn rate in dataset: {df.filter(F.col('churned') == 1).count() / df.count():.2%}")

    write_to_feature_store(df, jdbc_url)

    spark.stop()
    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()