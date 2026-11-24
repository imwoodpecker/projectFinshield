"""DuckDB database layer for FinShield."""
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging
import os

from src.common.config import get_config
from src.common.models import Transaction, FMSAlert, TMSAlert, GovernmentWatchlistEntry

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Manages DuckDB connections and operations."""

    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        # Use in-memory DB for testing to avoid concurrent access issues
        if db_path == ":memory:" or os.getenv("FINSHIELD_TEST_MODE") == "1":
            self.db_path = ":memory:"
        else:
            self.db_path = db_path or self.config.get("ingestion.duckdb.path", "data/finshield.duckdb")
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            # Configure for performance (skip for in-memory)
            if self.db_path != ":memory:":
                self._conn.execute(f"PRAGMA memory_limit='{self.config.get('ingestion.duckdb.memory_limit', '4GB')}'")
                self._conn.execute(f"PRAGMA threads={self.config.get('ingestion.duckdb.threads', 4)}")
        return self._conn

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()

        # Transactions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                amount DOUBLE NOT NULL,
                currency VARCHAR DEFAULT 'INR',
                channel VARCHAR NOT NULL,
                txn_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                beneficiary_account_id VARCHAR,
                beneficiary_name VARCHAR,
                device_id VARCHAR,
                ip_address VARCHAR,
                phone_number VARCHAR,
                location VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_beneficiary ON transactions(beneficiary_account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_device ON transactions(device_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_ip ON transactions(ip_address)")

        # FMS Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fms_alerts (
                alert_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                alert_type VARCHAR NOT NULL,
                severity VARCHAR NOT NULL,
                description VARCHAR,
                source_system VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                raw_payload JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fms_account ON fms_alerts(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fms_timestamp ON fms_alerts(timestamp)")

        # TMS Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tms_alerts (
                alert_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                rule_id VARCHAR NOT NULL,
                rule_name VARCHAR NOT NULL,
                risk_score DOUBLE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                transaction_ids JSON,
                raw_payload JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tms_account ON tms_alerts(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tms_timestamp ON tms_alerts(timestamp)")

        # Government Watchlists table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                watchlist_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                account_id VARCHAR,
                entity_name VARCHAR,
                entity_type VARCHAR NOT NULL,
                identifiers JSON,
                listed_date TIMESTAMP NOT NULL,
                reason VARCHAR,
                severity VARCHAR,
                raw_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_account ON watchlists(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_source ON watchlists(source)")

        # Account Profiles table (90-day baselines)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS account_profiles (
                account_id VARCHAR PRIMARY KEY,
                window_start TIMESTAMP NOT NULL,
                window_end TIMESTAMP NOT NULL,
                avg_txn_amount DOUBLE,
                std_txn_amount DOUBLE,
                txn_count INTEGER,
                avg_daily_txn_count DOUBLE,
                preferred_hours JSON,
                preferred_channels JSON,
                top_beneficiaries JSON,
                dormancy_periods JSON,
                velocity_score DOUBLE DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Anomaly Scores table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_scores (
                account_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                isolation_forest_score DOUBLE,
                ecod_score DOUBLE,
                loda_score DOUBLE,
                ensemble_score DOUBLE,
                is_anomaly BOOLEAN,
                features_used JSON,
                PRIMARY KEY (account_id, timestamp)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_account ON anomaly_scores(account_id)")

        # Graph Metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_metrics (
                account_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                degree_centrality DOUBLE,
                betweenness_centrality DOUBLE,
                closeness_centrality DOUBLE,
                pagerank DOUBLE,
                cycle_count INTEGER,
                max_cycle_length INTEGER,
                shared_device_count INTEGER,
                shared_ip_count INTEGER,
                kingpin_score DOUBLE,
                mule_network_score DOUBLE,
                PRIMARY KEY (account_id, timestamp)
            )
        """)

        # Risk Assessments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                assessment_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                ml_score DOUBLE,
                graph_score DOUBLE,
                regulatory_score DOUBLE,
                composite_score DOUBLE,
                risk_tier VARCHAR,
                classification VARCHAR,
                mule_probability DOUBLE,
                suspicious_probability DOUBLE,
                legitimate_probability DOUBLE,
                shap_explanation JSON,
                watchlist_matches JSON,
                recommended_action VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_account ON risk_assessments(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_composite ON risk_assessments(composite_score DESC)")

        # SAR Reports table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sar_reports (
                sar_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                risk_assessment_id VARCHAR NOT NULL,
                generated_at TIMESTAMP NOT NULL,
                filed_at TIMESTAMP,
                status VARCHAR DEFAULT 'DRAFT',
                narrative TEXT,
                transaction_summary JSON,
                graph_evidence_summary JSON,
                regulatory_references JSON,
                recommended_action VARCHAR,
                compliance_officer VARCHAR,
                fiud_reference VARCHAR
            )
        """)

        # Cases table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                risk_assessment_id VARCHAR NOT NULL,
                tier VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_to VARCHAR,
                sla_deadline TIMESTAMP,
                resolved_at TIMESTAMP,
                feedback VARCHAR,
                feedback_notes VARCHAR,
                analyst_id VARCHAR
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_case_account ON cases(account_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_case_status ON cases(status)")

        # Model Feedback table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_feedback (
                feedback_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                case_id VARCHAR NOT NULL,
                predicted_label VARCHAR NOT NULL,
                true_label VARCHAR NOT NULL,
                features JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        logger.info("Database schema initialized successfully")

    # Transaction operations
    def insert_transaction(self, txn: Transaction) -> None:
        """Insert a single transaction."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO transactions (transaction_id, account_id, amount, currency, channel, txn_type, timestamp,
                beneficiary_account_id, beneficiary_name, device_id, ip_address, phone_number, location, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            txn.transaction_id, txn.account_id, txn.amount, txn.currency,
            txn.channel.value, txn.txn_type.value, txn.timestamp,
            txn.beneficiary_account_id, txn.beneficiary_name,
            txn.device_id, txn.ip_address, txn.phone_number,
            txn.location, str(txn.metadata)
        ])

    def insert_transactions_batch(self, transactions: List[Transaction]) -> None:
        """Bulk insert transactions."""
        conn = self._get_connection()
        data = [
            [
                t.transaction_id, t.account_id, t.amount, t.currency,
                t.channel.value, t.txn_type.value, t.timestamp,
                t.beneficiary_account_id, t.beneficiary_name,
                t.device_id, t.ip_address, t.phone_number,
                t.location, str(t.metadata)
            ]
            for t in transactions
        ]
        conn.executemany("""
            INSERT INTO transactions (transaction_id, account_id, amount, currency, channel, txn_type, timestamp,
                beneficiary_account_id, beneficiary_name, device_id, ip_address, phone_number, location, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        logger.info(f"Inserted {len(transactions)} transactions")

    def get_account_transactions(self, account_id: str, days: int = 90) -> pd.DataFrame:
        """Get transactions for an account within the last N days."""
        conn = self._get_connection()
        return conn.execute(f"""
            SELECT * FROM transactions
            WHERE account_id = ? AND timestamp >= CURRENT_DATE - INTERVAL '{days}' DAYS
            ORDER BY timestamp
        """, [account_id]).fetchdf()

    def get_recent_transactions(self, hours: int = 1) -> pd.DataFrame:
        """Get recent transactions across all accounts."""
        conn = self._get_connection()
        return conn.execute(f"""
            SELECT * FROM transactions
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '{hours}' HOURS
            ORDER BY timestamp DESC
        """).fetchdf()

    def get_velocity_stats(self, account_id: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get transaction velocity statistics for an account."""
        conn = self._get_connection()
        result = conn.execute(f"""
            SELECT
                COUNT(*) as txn_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(timestamp) as first_txn,
                MAX(timestamp) as last_txn
            FROM transactions
            WHERE account_id = ? AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '{window_minutes}' MINUTES
        """, [account_id]).fetchone()

        if result and result[0] > 0:
            return {
                "txn_count": result[0],
                "total_amount": result[1],
                "avg_amount": result[2],
                "first_txn": result[3],
                "last_txn": result[4],
                "velocity_per_minute": result[0] / window_minutes
            }
        return {"txn_count": 0, "total_amount": 0, "avg_amount": 0, "velocity_per_minute": 0}

    # Alert operations
    def insert_fms_alert(self, alert: FMSAlert) -> None:
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO fms_alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            alert.alert_id, alert.account_id, alert.alert_type,
            alert.severity, alert.description, alert.source_system,
            alert.timestamp, str(alert.raw_payload)
        ])

    def insert_tms_alert(self, alert: TMSAlert) -> None:
        conn = self._get_connection()
        import json
        conn.execute("""
            INSERT INTO tms_alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            alert.alert_id, alert.account_id, alert.rule_id,
            alert.rule_name, alert.risk_score, alert.timestamp,
            json.dumps(alert.transaction_ids), json.dumps(alert.raw_payload)
        ])

    # Watchlist operations
    def insert_watchlist_entry(self, entry: GovernmentWatchlistEntry) -> None:
        conn = self._get_connection()
        import json
        conn.execute("""
            INSERT INTO watchlists VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            entry.watchlist_id, entry.source, entry.account_id,
            entry.entity_name, entry.entity_type, json.dumps(entry.identifiers),
            entry.listed_date, entry.reason, entry.severity,
            json.dumps(entry.raw_data)
        ])

    def is_account_watchlisted(self, account_id: str) -> List[Dict[str, Any]]:
        """Check if account is on any watchlist."""
        conn = self._get_connection()
        results = conn.execute("""
            SELECT source, entity_name, reason, severity, listed_date FROM watchlists
            WHERE account_id = ?
        """, [account_id]).fetchall()
        return [
            {"source": r[0], "entity_name": r[1], "reason": r[2], "severity": r[3], "listed_date": r[4]}
            for r in results
        ]

    # Profile operations
    def upsert_account_profile(self, profile: 'AccountProfile') -> None:
        """Insert or update account behavioural profile."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO account_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (account_id) DO UPDATE SET
                window_start = EXCLUDED.window_start,
                window_end = EXCLUDED.window_end,
                avg_txn_amount = EXCLUDED.avg_txn_amount,
                std_txn_amount = EXCLUDED.std_txn_amount,
                txn_count = EXCLUDED.txn_count,
                avg_daily_txn_count = EXCLUDED.avg_daily_txn_count,
                preferred_hours = EXCLUDED.preferred_hours,
                preferred_channels = EXCLUDED.preferred_channels,
                top_beneficiaries = EXCLUDED.top_beneficiaries,
                dormancy_periods = EXCLUDED.dormancy_periods,
                velocity_score = EXCLUDED.velocity_score,
                last_updated = CURRENT_TIMESTAMP
        """, [
            profile.account_id, profile.window_start, profile.window_end,
            profile.avg_txn_amount, profile.std_txn_amount, profile.txn_count,
            profile.avg_daily_txn_count, str(profile.preferred_hours),
            str([c.value for c in profile.preferred_channels]),
            str(profile.top_beneficiaries), str(profile.dormancy_periods),
            profile.velocity_score
        ])

    def get_account_profile(self, account_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        result = conn.execute("""
            SELECT * FROM account_profiles WHERE account_id = ?
        """, [account_id]).fetchone()
        if result:
            cols = [desc[0] for desc in conn.description]
            return dict(zip(cols, result))
        return None

    # Anomaly scores
    def insert_anomaly_score(self, score: 'AnomalyScore') -> None:
        conn = self._get_connection()
        import json
        conn.execute("""
            INSERT INTO anomaly_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_id, timestamp) DO UPDATE SET
                isolation_forest_score = EXCLUDED.isolation_forest_score,
                ecod_score = EXCLUDED.ecod_score,
                loda_score = EXCLUDED.loda_score,
                ensemble_score = EXCLUDED.ensemble_score,
                is_anomaly = EXCLUDED.is_anomaly,
                features_used = EXCLUDED.features_used
        """, [
            score.account_id, score.timestamp,
            score.isolation_forest_score, score.ecod_score,
            score.loda_score, score.ensemble_score,
            score.is_anomaly, json.dumps(score.features_used)
        ])

    # Graph metrics
    def insert_graph_metrics(self, metrics: 'GraphMetrics') -> None:
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO graph_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_id, timestamp) DO UPDATE SET
                degree_centrality = EXCLUDED.degree_centrality,
                betweenness_centrality = EXCLUDED.betweenness_centrality,
                closeness_centrality = EXCLUDED.closeness_centrality,
                pagerank = EXCLUDED.pagerank,
                cycle_count = EXCLUDED.cycle_count,
                max_cycle_length = EXCLUDED.max_cycle_length,
                shared_device_count = EXCLUDED.shared_device_count,
                shared_ip_count = EXCLUDED.shared_ip_count,
                kingpin_score = EXCLUDED.kingpin_score,
                mule_network_score = EXCLUDED.mule_network_score
        """, [
            metrics.account_id, metrics.timestamp,
            metrics.degree_centrality, metrics.betweenness_centrality,
            metrics.closeness_centrality, metrics.pagerank,
            metrics.cycle_count, metrics.max_cycle_length,
            metrics.shared_device_count, metrics.shared_ip_count,
            metrics.kingpin_score, metrics.mule_network_score
        ])

    # Risk assessments
    def insert_risk_assessment(self, assessment: 'RiskAssessment') -> None:
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO risk_assessments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            assessment.assessment_id, assessment.account_id, assessment.timestamp,
            assessment.ml_score, assessment.graph_score, assessment.regulatory_score,
            assessment.composite_score, assessment.risk_tier.value,
            assessment.classification, assessment.mule_probability,
            assessment.suspicious_probability, assessment.legitimate_probability,
            str(assessment.shap_explanation) if assessment.shap_explanation else '{}',
            str(assessment.watchlist_matches), assessment.recommended_action
        ])

    def get_latest_risk_assessment(self, account_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        result = conn.execute("""
            SELECT * FROM risk_assessments
            WHERE account_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, [account_id]).fetchone()
        if result:
            cols = [desc[0] for desc in conn.description]
            return dict(zip(cols, result))
        return None

    def get_top_risk_accounts(self, limit: int = 50) -> pd.DataFrame:
        conn = self._get_connection()
        return conn.execute("""
            SELECT account_id, composite_score, risk_tier, classification, timestamp
            FROM risk_assessments
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM risk_assessments ra2 WHERE ra2.account_id = risk_assessments.account_id
            )
            ORDER BY composite_score DESC
            LIMIT ?
        """, [limit]).fetchdf()

    # Cases
    def insert_case(self, case: 'Case') -> None:
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            case.case_id, case.account_id, case.risk_assessment_id,
            case.tier.value, case.status, case.created_at,
            case.assigned_to, case.sla_deadline, case.resolved_at,
            case.feedback.value if case.feedback else None,
            case.feedback_notes, case.analyst_id
        ])

    def update_case(self, case_id: str, **kwargs) -> None:
        conn = self._get_connection()
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [case_id]
        conn.execute(f"UPDATE cases SET {set_clause} WHERE case_id = ?", values)

    def get_open_cases(self) -> pd.DataFrame:
        conn = self._get_connection()
        return conn.execute("""
            SELECT * FROM cases WHERE status IN ('OPEN', 'IN_REVIEW')
            ORDER BY sla_deadline
        """).fetchdf()

    # Feedback
    def insert_feedback(self, feedback: 'ModelFeedback') -> None:
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO model_feedback VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            feedback.feedback_id, feedback.account_id, feedback.case_id,
            feedback.predicted_label, feedback.true_label.value,
            str(feedback.features)
        ])

    def get_feedback_for_retraining(self, min_samples: int = 100) -> pd.DataFrame:
        conn = self._get_connection()
        return conn.execute("""
            SELECT * FROM model_feedback
            WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
        """).fetchdf()

    def save_freeze(self, freeze_result: Dict[str, Any]) -> None:
        """Save account freeze record."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS account_freezes (
                freeze_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                reason VARCHAR,
                frozen_at TIMESTAMP NOT NULL,
                frozen_by VARCHAR,
                api_response JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        import json
        conn.execute("""
            INSERT INTO account_freezes VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            freeze_result["freeze_id"],
            freeze_result["account_id"],
            freeze_result["status"],
            freeze_result["reason"],
            freeze_result["frozen_at"],
            freeze_result["frozen_by"],
            json.dumps(freeze_result["api_response"])
        ])
        logger.info(f"Saved freeze record for account {freeze_result['account_id']}")

    def save_escalation(self, escalation: Dict[str, Any]) -> None:
        """Save escalation record."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                escalation_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                risk_assessment JSON,
                reasoning VARCHAR,
                escalated_at TIMESTAMP NOT NULL,
                escalated_to VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        import uuid
        import json
        escalation_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            escalation_id,
            escalation["account_id"],
            json.dumps(escalation["risk_assessment"]),
            escalation["reasoning"],
            escalation["escalated_at"],
            escalation["escalated_to"]
        ])
        logger.info(f"Saved escalation for account {escalation['account_id']}")

    def insert_sar_report(self, sar: 'SARReport') -> None:
        """Insert SAR report."""
        conn = self._get_connection()
        import json
        conn.execute("""
            INSERT INTO sar_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            sar.sar_id, sar.account_id, sar.risk_assessment_id,
            sar.generated_at, sar.filed_at, sar.status,
            sar.narrative, json.dumps(sar.transaction_summary),
            json.dumps(sar.graph_evidence_summary), json.dumps(sar.regulatory_references),
            sar.recommended_action, sar.compliance_officer,
            sar.fiud_reference
        ])
        logger.info(f"Inserted SAR report {sar.sar_id}")

    def log_agent_action(self, log_entry: Dict[str, Any]) -> None:
        """Log agent action to database."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_actions (
                action_id VARCHAR PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                case_id VARCHAR,
                action VARCHAR NOT NULL,
                reasoning VARCHAR,
                risk_assessment JSON,
                timestamp TIMESTAMP NOT NULL,
                agent_version VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        import uuid
        import json
        action_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO agent_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            action_id,
            log_entry["account_id"],
            log_entry.get("case_id"),
            log_entry["action"],
            log_entry["reasoning"],
            json.dumps(log_entry.get("risk_assessment", {})),
            log_entry["timestamp"],
            log_entry.get("agent_version", "1.0")
        ])
        logger.info(f"Logged agent action {log_entry['action']} for account {log_entry['account_id']}")

    def get_all_account_ids(self) -> List[str]:
        """Get all unique account IDs from transactions."""
        conn = self._get_connection()
        results = conn.execute("SELECT DISTINCT account_id FROM transactions").fetchall()
        return [r[0] for r in results]

    def find_accounts_by_device(self, device_id: str) -> List[str]:
        """Find all accounts that used a specific device."""
        conn = self._get_connection()
        results = conn.execute("SELECT DISTINCT account_id FROM transactions WHERE device_id = ?", [device_id]).fetchall()
        return [r[0] for r in results]

    def find_accounts_by_ip(self, ip_address: str) -> List[str]:
        """Find all accounts that used a specific IP address."""
        conn = self._get_connection()
        results = conn.execute("SELECT DISTINCT account_id FROM transactions WHERE ip_address = ?", [ip_address]).fetchall()
        return [r[0] for r in results]

    def find_accounts_by_phone(self, phone_number: str) -> List[str]:
        """Find all accounts that used a specific phone number."""
        conn = self._get_connection()
        results = conn.execute("SELECT DISTINCT account_id FROM transactions WHERE phone_number = ?", [phone_number]).fetchall()
        return [r[0] for r in results]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global instance
_db_manager: Optional[DuckDBManager] = None


def get_db() -> DuckDBManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DuckDBManager()
    return _db_manager