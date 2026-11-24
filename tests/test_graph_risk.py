"""Unit tests for graph and risk layers."""
import pytest
import networkx as nx
from datetime import datetime
from src.graph.analysis import GraphAnalyzer
from src.risk.engine import RiskEngine, WatchlistManager
from src.ingestion.synthetic_data import SyntheticDataGenerator
from src.ingestion.database import get_db
from src.common.models import (
    Transaction, TransactionChannel, TransactionType,
    GovernmentWatchlistEntry, RiskTier
)


class TestGraphAnalyzer:
    """Tests for graph analysis."""

    @pytest.fixture
    def built_graph(self):
        """Create a graph with synthetic data."""
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 100,
                    "num_transactions": 5000,
                    "fraud_ratio": 0.2
                }
            }
        }
        generator = SyntheticDataGenerator(config=config)
        accounts = generator.generate_accounts()
        transactions = generator.generate_transactions()

        all_accounts = generator.all_accounts

        analyzer = GraphAnalyzer()
        analyzer.build_graph(all_accounts, transactions)

        return analyzer, all_accounts, generator.account_profiles

    def test_graph_construction(self, built_graph):
        """Test graph is built correctly."""
        analyzer, accounts, profiles = built_graph

        assert analyzer.graph.number_of_nodes() > len(accounts)  # Accounts + entities
        assert analyzer.graph.number_of_edges() > 0
        assert len(analyzer.account_nodes) == len(accounts)

    def test_find_cycles(self, built_graph):
        """Test cycle detection."""
        analyzer, accounts, profiles = built_graph

        cycles = analyzer.find_cycles()

        # Should find some cycles in mule network data
        assert isinstance(cycles, list)
        for cycle in cycles:
            assert len(cycle) >= analyzer.min_cycle_length
            assert len(cycle) <= analyzer.max_cycle_length

    def test_find_cycles_for_account(self, built_graph):
        """Test cycle detection for specific account."""
        analyzer, accounts, profiles = built_graph

        # Test with a mule account (more likely to have cycles)
        mule_accounts = [a for a in accounts if 'MULE' in a]
        if mule_accounts:
            cycles = analyzer.find_cycles(mule_accounts[0])
            assert isinstance(cycles, list)

    def test_find_shared_entities(self, built_graph):
        """Test shared entity detection."""
        analyzer, accounts, profiles = built_graph

        # Test with an account that should have shared entities
        test_account = accounts[0]
        shared = analyzer.find_shared_entities(test_account)

        assert isinstance(shared, dict)
        for entity_type, shared_accounts in shared.items():
            assert entity_type in ['device_id', 'ip_address', 'phone_number']
            assert isinstance(shared_accounts, list)

    def test_compute_centrality_metrics(self, built_graph):
        """Test centrality computation."""
        analyzer, accounts, profiles = built_graph

        metrics = analyzer.compute_centrality_metrics()

        assert len(metrics) == len(accounts)
        for acc, metric in metrics.items():
            assert 'in_degree_centrality' in metric
            assert 'out_degree_centrality' in metric
            assert 'betweenness_centrality' in metric
            assert 'closeness_centrality' in metric
            assert 'pagerank' in metric
            assert all(0 <= v <= 1 for v in metric.values())

    def test_detect_kingpins(self, built_graph):
        """Test kingpin detection."""
        analyzer, accounts, profiles = built_graph

        kingpins = analyzer.detect_kingpins()

        assert isinstance(kingpins, list)
        for kp in kingpins:
            assert 'account_id' in kp
            assert 'in_degree' in kp
            assert 'kingpin_score' in kp
            assert 'senders' in kp
            assert kp['in_degree'] >= analyzer.kingpin_min_in_degree

    def test_detect_mule_networks(self, built_graph):
        """Test mule network detection."""
        analyzer, accounts, profiles = built_graph

        networks = analyzer.detect_mule_networks()

        assert isinstance(networks, list)
        for net in networks:
            assert 'accounts' in net
            assert 'size' in net
            assert 'density' in net
            assert 'mule_network_score' in net
            assert net['size'] >= 3
            assert 0 <= net['density'] <= 1

    def test_compute_account_metrics(self, built_graph):
        """Test comprehensive account metrics."""
        analyzer, accounts, profiles = built_graph

        test_account = accounts[0]
        metrics = analyzer.compute_account_metrics(test_account)

        assert metrics.account_id == test_account
        assert metrics.cycle_count >= 0
        assert metrics.shared_device_count >= 0
        assert metrics.shared_ip_count >= 0
        assert 0 <= metrics.kingpin_score <= 100
        assert 0 <= metrics.mule_network_score <= 100

    def test_get_graph_for_visualization(self, built_graph):
        """Test visualization data export."""
        analyzer, accounts, profiles = built_graph

        viz_data = analyzer.get_graph_for_visualization(max_nodes=50)

        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert len(viz_data['nodes']) <= 50
        assert all('id' in n and 'color' in n for n in viz_data['nodes'])


class TestRiskEngine:
    """Tests for risk scoring engine."""

    @pytest.fixture
    def risk_engine(self):
        """Create risk engine with test data."""
        # Generate and load synthetic data
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 100,
                    "num_transactions": 5000,
                    "fraud_ratio": 0.2
                }
            }
        }
        generator = SyntheticDataGenerator(config=config)
        accounts = generator.generate_accounts()
        transactions = generator.generate_transactions()

        db = get_db()
        db.insert_transactions_batch(transactions)

        # Train anomaly detector
        from src.detection.anomaly import AnomalyDetector
        all_accounts = generator.all_accounts
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(all_accounts)

        # Build graph
        from src.graph.analysis import GraphAnalyzer
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.build_graph(all_accounts, transactions)

        # Train classifier
        from src.detection.classifier import MuleAccountClassifier
        classifier = MuleAccountClassifier()
        labels = {}
        for acc in all_accounts:
            if 'MULE' in acc or 'KING' in acc:
                labels[acc] = 'MULE'
            elif 'SUS' in acc:
                labels[acc] = 'SUSPICIOUS'
            else:
                labels[acc] = 'LEGITIMATE'
        classifier.fit(all_accounts, labels)

        # Create risk engine
        engine = RiskEngine()
        return engine, all_accounts

    def test_compute_ml_score(self, risk_engine):
        """Test ML score computation."""
        engine, accounts = risk_engine

        score = engine.compute_ml_score(accounts[0])
        assert 0 <= score <= 100

    def test_compute_graph_score(self, risk_engine):
        """Test graph score computation."""
        engine, accounts = risk_engine

        score = engine.compute_graph_score(accounts[0])
        assert 0 <= score <= 100

    def test_compute_regulatory_score(self, risk_engine):
        """Test regulatory score computation."""
        engine, accounts = risk_engine

        # Should be 0 for accounts not on watchlist
        score = engine.compute_regulatory_score(accounts[0])
        assert 0 <= score <= 100

    def test_compute_composite_score(self, risk_engine):
        """Test composite score computation."""
        engine, accounts = risk_engine

        composite = engine.compute_composite_score(50, 30, 10, 20)
        expected = 50 * 0.4 + 30 * 0.3 + 10 * 0.2 + 20 * 0.1
        assert abs(composite - expected) < 0.01

    def test_determine_risk_tier(self, risk_engine):
        """Test risk tier determination."""
        engine, accounts = risk_engine

        assert engine.determine_risk_tier(85) == RiskTier.ACT
        assert engine.determine_risk_tier(65) == RiskTier.ALERT
        assert engine.determine_risk_tier(35) == RiskTier.WATCH
        assert engine.determine_risk_tier(15) == RiskTier.WATCH  # Minimum tier is WATCH

    def test_assess_account(self, risk_engine):
        """Test full account assessment."""
        engine, accounts = risk_engine

        assessment = engine.assess_account(accounts[0])

        assert assessment.account_id == accounts[0]
        assert 0 <= assessment.composite_score <= 100
        assert assessment.risk_tier in RiskTier
        assert assessment.classification in ['MULE', 'SUSPICIOUS', 'LEGITIMATE', 'UNKNOWN']
        assert 0 <= assessment.mule_probability <= 100
        assert 0 <= assessment.suspicious_probability <= 100
        assert 0 <= assessment.legitimate_probability <= 100

    def test_get_high_risk_accounts(self, risk_engine):
        """Test high-risk account retrieval."""
        engine, accounts = risk_engine

        high_risk = engine.get_high_risk_accounts(RiskTier.ALERT, limit=5)

        assert len(high_risk) <= 5
        for assessment in high_risk:
            assert assessment.risk_tier in [RiskTier.ALERT, RiskTier.ACT]
            assert assessment.composite_score >= engine.thresholds['alert']

    def test_explain_assessment(self, risk_engine):
        """Test assessment explanation."""
        engine, accounts = risk_engine

        explanation = engine.explain_assessment(accounts[0])

        assert 'account_id' in explanation
        assert 'composite_score' in explanation
        assert 'risk_tier' in explanation
        assert 'reasons' in explanation
        assert isinstance(explanation['reasons'], list)
        assert 'component_scores' in explanation

    def test_get_risk_summary(self, risk_engine):
        """Test portfolio risk summary."""
        engine, accounts = risk_engine

        summary = engine.get_risk_summary()

        assert 'total_accounts' in summary
        assert 'tier_distribution' in summary
        assert 'average_scores' in summary
        assert 'top_risk_accounts' in summary
        # total_accounts may include accounts from other tests, so just check it's >= our test accounts
        assert summary['total_accounts'] >= len(accounts)


class TestWatchlistManager:
    """Tests for watchlist management."""

    @pytest.fixture
    def watchlist_mgr(self):
        """Create watchlist manager."""
        return WatchlistManager()

    def test_ingest_watchlist(self, watchlist_mgr):
        """Test watchlist ingestion."""
        entries = [
            {
                'watchlist_id': 'WL001',
                'account_id': 'ACC_TEST_1',
                'entity_name': 'Test Entity 1',
                'entity_type': 'INDIVIDUAL',
                'identifiers': {'pan': 'ABCDE1234F'},
                'reason': 'Test reason',
                'severity': 'HIGH'
            },
            {
                'watchlist_id': 'WL002',
                'account_id': 'ACC_TEST_2',
                'entity_name': 'Test Entity 2',
                'entity_type': 'ENTITY',
                'identifiers': {'cin': 'U12345DL2020PTC123456'},
                'reason': 'Test reason 2',
                'severity': 'CRITICAL'
            }
        ]

        count = watchlist_mgr.ingest_watchlist("TEST_SOURCE", entries)
        assert count == 2

    def test_cross_channel_identity_resolution(self, watchlist_mgr):
        """Test cross-channel identity resolution."""
        # Setup: create accounts with shared entities
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 50,
                    "num_transactions": 2000,
                    "fraud_ratio": 0.2
                }
            }
        }
        generator = SyntheticDataGenerator(config=config)
        accounts = generator.generate_accounts()
        transactions = generator.generate_transactions()

        db = get_db()
        db.insert_transactions_batch(transactions)

        # Test resolution
        all_accounts = generator.all_accounts
        if all_accounts:
            related = watchlist_mgr.cross_channel_identity_resolution(all_accounts[0])
            assert isinstance(related, list)

    def test_get_watchlist_stats(self, watchlist_mgr):
        """Test watchlist statistics."""
        stats = watchlist_mgr.get_watchlist_stats()

        assert 'total_entries' in stats
        assert 'by_source_severity' in stats
        assert 'sources' in stats
        assert isinstance(stats['total_entries'], int)