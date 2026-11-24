"""Risk Engine - Composite risk scoring and regulatory integration."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import json

from src.common.models import (
    RiskAssessment, RiskTier, Transaction, GovernmentWatchlistEntry,
    FMSAlert, TMSAlert, AccountProfile, GraphMetrics, AnomalyScore
)
from src.ingestion.database import get_db
from src.common.config import get_config
from src.detection.classifier import MuleAccountClassifier
from src.graph.analysis import GraphAnalyzer

logger = logging.getLogger(__name__)


class RiskEngine:
    """Composite risk scoring engine combining ML, graph, and regulatory signals."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.classifier = MuleAccountClassifier()
        self.graph_analyzer = GraphAnalyzer()

        # Risk tier thresholds
        self.thresholds = self.config.get("risk_engine.thresholds", {
            "watch": 30,
            "alert": 60,
            "act": 80
        })

        # Weights for composite score
        self.weights = self.config.get("risk_engine.weights", {
            "ml": 0.4,
            "graph": 0.3,
            "regulatory": 0.2,
            "anomaly": 0.1
        })

    def compute_ml_score(self, account_id: str) -> float:
        """Get ML score from classifier."""
        try:
            assessment = self.classifier.predict(account_id)
            return assessment.ml_score
        except Exception as e:
            logger.warning(f"ML score error for {account_id}: {e}")
            return 0.0

    def compute_graph_score(self, account_id: str) -> float:
        """Get graph-based risk score."""
        try:
            metrics = self.graph_analyzer.get_account_graph_metrics(account_id)

            # Weighted combination of graph metrics
            score = (
                metrics.get('kingpin_score', 0) * 0.3 +
                metrics.get('mule_network_score', 0) * 0.3 +
                min(metrics.get('cycle_count', 0) * 5, 30) * 0.2 +
                min(metrics.get('shared_device_count', 0) * 5, 20) * 0.1 +
                min(metrics.get('shared_ip_count', 0) * 5, 20) * 0.1
            )

            return min(score, 100.0)
        except Exception as e:
            logger.warning(f"Graph score error for {account_id}: {e}")
            return 0.0

    def compute_regulatory_score(self, account_id: str) -> float:
        """Get regulatory/watchlist risk score."""
        try:
            matches = self.db.is_account_watchlisted(account_id)
            if not matches:
                return 0.0

            severity_weights = {'LOW': 10, 'MEDIUM': 30, 'HIGH': 60, 'CRITICAL': 90}
            max_score = 0
            for match in matches:
                score = severity_weights.get(match.get('severity', 'LOW'), 10)
                max_score = max(max_score, score)

            return min(max_score, 100.0)
        except Exception as e:
            logger.warning(f"Regulatory score error for {account_id}: {e}")
            return 0.0

    def compute_anomaly_score(self, account_id: str) -> float:
        """Get anomaly detection score."""
        try:
            assessment = self.db.get_latest_risk_assessment(account_id)
            if assessment and 'anomaly_ensemble' in assessment:
                return assessment['anomaly_ensemble']
            return 0.0
        except Exception as e:
            logger.warning(f"Anomaly score error for {account_id}: {e}")
            return 0.0

    def compute_composite_score(self, ml_score: float, graph_score: float,
                                regulatory_score: float, anomaly_score: float) -> float:
        """Compute weighted composite risk score."""
        return (
            ml_score * self.weights.get('ml', 0.4) +
            graph_score * self.weights.get('graph', 0.3) +
            regulatory_score * self.weights.get('regulatory', 0.2) +
            anomaly_score * self.weights.get('anomaly', 0.1)
        )

    def determine_risk_tier(self, composite_score: float) -> RiskTier:
        """Determine risk tier from composite score."""
        if composite_score >= self.thresholds.get('act', 80):
            return RiskTier.ACT
        elif composite_score >= self.thresholds.get('alert', 60):
            return RiskTier.ALERT
        elif composite_score >= self.thresholds.get('watch', 30):
            return RiskTier.WATCH
        else:
            return RiskTier.WATCH  # Minimum tier is WATCH (no NORMAL tier in enum)

    def assess_account(self, account_id: str) -> RiskAssessment:
        """Complete risk assessment for an account."""
        logger.info(f"Assessing account {account_id}...")

        # Get individual scores
        ml_score = self.compute_ml_score(account_id)
        graph_score = self.compute_graph_score(account_id)
        regulatory_score = self.compute_regulatory_score(account_id)
        anomaly_score = self.compute_anomaly_score(account_id)

        # Composite score
        composite_score = self.compute_composite_score(
            ml_score, graph_score, regulatory_score, anomaly_score
        )

        # Risk tier
        risk_tier = self.determine_risk_tier(composite_score)

        # Get ML classification and probabilities
        ml_details = {}
        try:
            assessment = self.classifier.predict(account_id)
            ml_details = {
                'classification': assessment.classification,
                'mule_probability': assessment.mule_probability,
                'suspicious_probability': assessment.suspicious_probability,
                'legitimate_probability': assessment.legitimate_probability,
                'shap_explanation': assessment.shap_explanation
            }
        except:
            ml_details = {}

        # Get graph details
        graph_details = {}
        try:
            graph_details = self.graph_analyzer.get_account_graph_metrics(account_id)
        except:
            pass

        # Get regulatory details
        regulatory_details = {}
        try:
            matches = self.db.is_account_watchlisted(account_id)
            regulatory_details = {
                'watchlist_matches': matches,
                'match_count': len(matches)
            }
        except:
            pass

        # Get anomaly details
        anomaly_details = {}
        try:
            assessment = self.db.get_latest_risk_assessment(account_id)
            if assessment:
                anomaly_details = {
                    'isolation_forest': assessment.get('isolation_forest_score', 0),
                    'ecod': assessment.get('ecod_score', 0),
                    'loda': assessment.get('loda_score', 0),
                    'ensemble': assessment.get('ensemble_score', 0)
                }
        except:
            pass

        # Determine recommended action based on risk tier and classification
        if risk_tier == RiskTier.ACT:
            recommended_action = "FREEZE_ACCOUNT"
        elif risk_tier == RiskTier.ALERT:
            recommended_action = "GENERATE_SAR"
        elif risk_tier == RiskTier.WATCH:
            recommended_action = "INVESTIGATE"
        else:
            recommended_action = "MONITOR"

        return RiskAssessment(
            account_id=account_id,
            timestamp=datetime.utcnow(),
            ml_score=ml_score,
            graph_score=graph_score,
            regulatory_score=regulatory_score,
            anomaly_score=anomaly_score,
            composite_score=composite_score,
            risk_tier=risk_tier,
            classification=ml_details.get('classification', 'UNKNOWN'),
            mule_probability=ml_details.get('mule_probability', 0),
            suspicious_probability=ml_details.get('suspicious_probability', 0),
            legitimate_probability=ml_details.get('legitimate_probability', 0),
            shap_explanation=ml_details.get('shap_explanation', {}),
            graph_details=graph_details,
            regulatory_details=regulatory_details,
            anomaly_details=anomaly_details,
            recommended_action=recommended_action
        )

    def assess_batch(self, account_ids: List[str]) -> List[RiskAssessment]:
        """Assess multiple accounts."""
        results = []
        for acc_id in account_ids:
            try:
                results.append(self.assess_account(acc_id))
            except Exception as e:
                logger.error(f"Error assessing {acc_id}: {e}")
        return results

    def get_high_risk_accounts(self, tier: RiskTier = RiskTier.ALERT, limit: int = 100) -> List[RiskAssessment]:
        """Get accounts at or above a risk tier."""
        all_accounts = self.db.get_all_account_ids()
        assessments = self.assess_batch(all_accounts)

        filtered = [a for a in assessments if self._tier_order(a.risk_tier) >= self._tier_order(tier)]
        filtered.sort(key=lambda x: x.composite_score, reverse=True)

        return filtered[:limit]

    def _tier_order(self, tier: RiskTier) -> int:
        order = {RiskTier.WATCH: 1, RiskTier.ALERT: 2, RiskTier.ACT: 3}
        return order.get(tier, 0)

    def save_assessment(self, assessment: RiskAssessment) -> None:
        """Save risk assessment to database."""
        self.db.insert_risk_assessment(assessment)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get portfolio-level risk summary."""
        all_accounts = self.db.get_all_account_ids()
        assessments = self.assess_batch(all_accounts)

        tier_counts = defaultdict(int)
        for a in assessments:
            tier_counts[a.risk_tier.value] += 1

        avg_scores = {}
        if assessments:
            avg_scores = {
                'ml': np.mean([a.ml_score for a in assessments]),
                'graph': np.mean([a.graph_score for a in assessments]),
                'regulatory': np.mean([a.regulatory_score for a in assessments]),
                'anomaly': np.mean([a.anomaly_score for a in assessments]),
                'composite': np.mean([a.composite_score for a in assessments])
            }

        top_risk = sorted(assessments, key=lambda x: x.composite_score, reverse=True)[:10]

        return {
            'total_accounts': len(all_accounts),
            'tier_distribution': dict(tier_counts),
            'average_scores': avg_scores,
            'top_risk_accounts': [
                {'account_id': a.account_id, 'composite_score': a.composite_score, 'tier': a.risk_tier.value}
                for a in top_risk
            ],
            'timestamp': datetime.utcnow().isoformat()
        }

    def explain_assessment(self, account_id: str) -> Dict[str, Any]:
        """Generate human-readable explanation for an assessment."""
        assessment = self.assess_account(account_id)

        reasons = []
        if assessment.ml_score > 50:
            reasons.append(f"ML model flags as {assessment.classification} ({assessment.ml_score:.0f}% risk)")
        if assessment.graph_score > 50:
            if assessment.graph_details.get('kingpin_score', 0) > 50:
                reasons.append(f"Kingpin account detected (score: {assessment.graph_details['kingpin_score']:.0f})")
            if assessment.graph_details.get('mule_network_score', 0) > 50:
                reasons.append(f"Part of mule network (score: {assessment.graph_details['mule_network_score']:.0f})")
            if assessment.graph_details.get('cycle_count', 0) > 0:
                reasons.append(f"Circular flows detected ({assessment.graph_details['cycle_count']} cycles)")
            if assessment.graph_details.get('shared_device_count', 0) > 0:
                reasons.append(f"Shares device with {assessment.graph_details['shared_device_count']} accounts")
        if assessment.regulatory_score > 0:
            reasons.append(f"Watchlist match: {assessment.regulatory_details.get('match_count', 0)} entries")
        if assessment.anomaly_score > 50:
            reasons.append(f"Anomalous behavior pattern (score: {assessment.anomaly_score:.0f})")

        top_shap = assessment.shap_explanation
        if top_shap:
            top_feature = list(top_shap.keys())[0]
            reasons.append(f"Key ML driver: {top_feature}")

        return {
            'account_id': account_id,
            'composite_score': assessment.composite_score,
            'risk_tier': assessment.risk_tier.value,
            'reasons': reasons,
            'component_scores': {
                'ml': assessment.ml_score,
                'graph': assessment.graph_score,
                'regulatory': assessment.regulatory_score,
                'anomaly': assessment.anomaly_score
            },
            'details': {
                'ml': {
                    'classification': assessment.classification,
                    'probabilities': {
                        'mule': assessment.mule_probability,
                        'suspicious': assessment.suspicious_probability,
                        'legitimate': assessment.legitimate_probability
                    },
                    'shap': assessment.shap_explanation
                },
                'graph': assessment.graph_details,
                'regulatory': assessment.regulatory_details,
                'anomaly': assessment.anomaly_details
            }
        }


class WatchlistManager:
    """Manage government watchlist ingestion and matching."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()

    def ingest_watchlist(self, source: str, entries: List[Dict[str, Any]]) -> int:
        """Bulk ingest watchlist entries from government sources."""
        count = 0
        for entry_data in entries:
            try:
                entry = GovernmentWatchlistEntry(
                    watchlist_id=entry_data.get('watchlist_id', ''),
                    source=source,
                    account_id=entry_data.get('account_id'),
                    entity_name=entry_data.get('entity_name'),
                    entity_type=entry_data.get('entity_type', 'INDIVIDUAL'),
                    identifiers=entry_data.get('identifiers', {}),
                    listed_date=entry_data.get('listed_date', datetime.utcnow()),
                    reason=entry_data.get('reason', ''),
                    severity=entry_data.get('severity', 'MEDIUM'),
                    raw_data=entry_data.get('raw_data', {})
                )
                self.db.insert_watchlist_entry(entry)
                count += 1
            except Exception as e:
                logger.error(f"Failed to ingest watchlist entry: {e}")

        logger.info(f"Ingested {count} watchlist entries from {source}")
        return count

    def load_from_file(self, source: str, filepath: str) -> int:
        """Load watchlist from JSON/CSV file."""
        if filepath.endswith('.json'):
            with open(filepath) as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.to_dict(orient='records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")

        return self.ingest_watchlist(source, data)

    def cross_channel_identity_resolution(self, account_id: str) -> List[str]:
        """Find related accounts across channels using shared identifiers."""
        related = set()

        # Get all transactions for this account
        df = self.db.get_account_transactions(account_id, days=180)

        if df.empty:
            return []

        # Collect all shared entities
        devices = df['device_id'].dropna().unique()
        ips = df['ip_address'].dropna().unique()
        phones = df['phone_number'].dropna().unique()

        # Find other accounts using same entities
        for device in devices:
            others = self.db.find_accounts_by_device(device)
            related.update(others)

        for ip in ips:
            others = self.db.find_accounts_by_ip(ip)
            related.update(others)

        for phone in phones:
            others = self.db.find_accounts_by_phone(phone)
            related.update(others)

        related.discard(account_id)
        return list(related)

    def get_watchlist_stats(self) -> Dict[str, Any]:
        """Get watchlist statistics."""
        conn = self.db._get_connection()
        stats = conn.execute("""
            SELECT source, severity, COUNT(*) as count
            FROM watchlists
            GROUP BY source, severity
        """).fetchdf()

        total = conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0]

        return {
            'total_entries': total,
            'by_source_severity': stats.to_dict(orient='records'),
            'sources': conn.execute("SELECT DISTINCT source FROM watchlists").fetchall()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = RiskEngine()

    # Test with some accounts
    db = get_db()
    accounts = db.get_all_account_ids()[:5]

    for acc in accounts:
        assessment = engine.assess_account(acc)
        print(f"{acc}: Score={assessment.composite_score:.1f}, Tier={assessment.risk_tier.value}")

    print("\nPortfolio Summary:")
    print(json.dumps(engine.get_risk_summary(), indent=2, default=str))