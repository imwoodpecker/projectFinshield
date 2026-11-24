"""Unit tests for detection layer."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.detection.features import FeatureExtractor, build_feature_matrix
from src.detection.anomaly import AnomalyDetector, StreamingAnomalyDetector
from src.detection.classifier import MuleAccountClassifier, train_full_pipeline
from src.ingestion.synthetic_data import SyntheticDataGenerator
from src.ingestion.database import get_db
from src.common.models import Transaction, TransactionChannel, TransactionType


class TestFeatureExtractor:
    """Tests for feature extraction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
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
        return generator.account_profiles, transactions, accounts

    def test_extract_basic_features(self, sample_data):
        """Test basic feature extraction."""
        profiles, transactions, accounts = sample_data
        extractor = FeatureExtractor()
        account_list = accounts['legitimate'][:5]

        features = extractor.extract_all_features(account_list, transactions)

        assert 'account_id' in features.columns
        assert len(features) == 5
        assert 'txn_count' in features.columns
        assert 'total_amount' in features.columns
        assert 'avg_amount' in features.columns

    def test_velocity_features(self, sample_data):
        """Test velocity feature extraction."""
        profiles, transactions, accounts = sample_data
        extractor = FeatureExtractor()
        account_list = accounts['legitimate'][:3]

        features = extractor.extract_all_features(account_list, transactions)

        assert 'txn_per_day' in features.columns
        assert 'max_txn_per_hour' in features.columns
        assert 'velocity_score' in features.columns

    def test_structuring_features(self, sample_data):
        """Test structuring/smurfing feature extraction."""
        profiles, transactions, accounts = sample_data
        extractor = FeatureExtractor()
        account_list = accounts['mule'][:3]

        features = extractor.extract_all_features(account_list, transactions)

        assert 'structuring_score' in features.columns
        assert 'round_amount_ratio' in features.columns

    def test_dormancy_features(self, sample_data):
        """Test dormancy break feature extraction."""
        profiles, transactions, accounts = sample_data
        extractor = FeatureExtractor()
        account_list = accounts['suspicious'][:3]

        features = extractor.extract_all_features(account_list, transactions)

        assert 'days_since_last_txn' in features.columns
        assert 'dormancy_break_flag' in features.columns


class TestAnomalyDetector:
    """Tests for anomaly detection."""

    @pytest.fixture
    def trained_detector(self):
        """Create and train anomaly detector on synthetic data."""
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 200,
                    "num_transactions": 10000,
                    "fraud_ratio": 0.2
                }
            }
        }
        generator = SyntheticDataGenerator(config=config)
        accounts_dict = generator.generate_accounts()
        transactions = generator.generate_transactions()

        accounts = generator.all_accounts

        detector = AnomalyDetector()
        detector.fit(accounts)

        return detector, accounts, transactions

    def test_fit_creates_models(self, trained_detector):
        """Test that fitting creates all ensemble models."""
        detector, accounts, transactions = trained_detector

        assert detector.is_fitted
        assert detector.isolation_forest is not None
        assert detector.ecod is not None
        assert detector.loda is not None

    def test_predict_single_account(self, trained_detector):
        """Test prediction for single account."""
        detector, accounts, transactions = trained_detector

        test_account = accounts[0]
        result = detector.predict(test_account)

        # Result is AnomalyScore object
        assert hasattr(result, 'isolation_forest_score')
        assert hasattr(result, 'ecod_score')
        assert hasattr(result, 'loda_score')
        assert hasattr(result, 'ensemble_score')

        # All scores should be between 0 and 100
        assert 0 <= result.isolation_forest_score <= 100 or result.isolation_forest_score == 0
        assert 0 <= result.ecod_score <= 100 or result.ecod_score == 0
        assert 0 <= result.loda_score <= 100 or result.loda_score == 0
        assert 0 <= result.ensemble_score <= 100

    def test_predict_batch(self, trained_detector):
        """Test batch prediction."""
        detector, accounts, transactions = trained_detector

        test_accounts = accounts[:5]
        results = detector.predict_batch(test_accounts)

        assert len(results) == 5
        for result in results:
            assert hasattr(result, 'ensemble_score')
            assert 0 <= result.ensemble_score <= 100

    def test_streaming_detector(self):
        """Test streaming anomaly detector."""
        streaming = StreamingAnomalyDetector()

        # Simulate streaming transactions
        for i in range(100):
            txn = Transaction(
                transaction_id=f"TXN{i}",
                account_id="ACC_TEST",
                amount=1000.0 + np.random.normal(0, 100),
                channel=TransactionChannel.UPI,
                txn_type=TransactionType.DEBIT,
                timestamp=datetime.utcnow()
            )
            score = streaming.update(txn)
            assert 0 <= score <= 100


class TestMuleAccountClassifier:
    """Tests for XGBoost classifier."""

    @pytest.fixture
    def sample_accounts(self):
        """Generate sample accounts with known labels."""
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 200,
                    "num_transactions": 10000,
                    "fraud_ratio": 0.2
                }
            }
        }
        generator = SyntheticDataGenerator(config=config)
        accounts_dict = generator.generate_accounts()
        transactions = generator.generate_transactions()

        accounts = generator.all_accounts

        # Insert data into database for feature extraction
        db = get_db()
        db.insert_transactions_batch(transactions)

        return accounts, accounts_dict

    def test_prepare_features(self, sample_accounts):
        """Test feature preparation."""
        accounts, account_groups = sample_accounts
        classifier = MuleAccountClassifier()

        feature_df, feature_cols = classifier.prepare_features(accounts[:10])

        assert len(feature_df) == 10
        assert len(feature_cols) > 20  # Should have many features
        assert 'anomaly_ensemble' in feature_cols
        # Graph features may not exist if no graph built, just check anomaly_ensemble

    def test_create_labels(self, sample_accounts):
        """Test label creation from account IDs."""
        accounts, account_groups = sample_accounts
        classifier = MuleAccountClassifier()

        labels = classifier.create_labels(accounts)

        assert len(labels) == len(accounts)
        for acc in accounts:
            if 'MULE' in acc or 'KING' in acc:
                assert labels[acc] == 'MULE'
            elif 'SUS' in acc:
                assert labels[acc] == 'SUSPICIOUS'
            else:
                assert labels[acc] == 'LEGITIMATE'

    def test_fit_and_predict(self, sample_accounts):
        """Test full training and prediction pipeline."""
        accounts, account_groups = sample_accounts
        classifier = MuleAccountClassifier()

        # Create labels
        labels = classifier.create_labels(accounts)

        # Train
        results = classifier.fit(accounts, labels)

        assert 'classification_report' in results
        assert 'roc_auc' in results
        assert classifier.is_fitted

        # Predict
        test_account = accounts[0]
        assessment = classifier.predict(test_account)

        assert assessment.account_id == test_account
        assert assessment.classification in ['MULE', 'SUSPICIOUS', 'LEGITIMATE']
        assert 0 <= assessment.mule_probability <= 100
        assert assessment.shap_explanation is not None

    def test_save_load(self, sample_accounts):
        """Test model persistence."""
        accounts, account_groups = sample_accounts

        # Ensure we have at least 2 classes for training
        labels = {}
        for acc in accounts:
            if 'MULE' in acc or 'KING' in acc:
                labels[acc] = 'MULE'
            elif 'SUS' in acc:
                labels[acc] = 'SUSPICIOUS'
            else:
                labels[acc] = 'LEGITIMATE'

        classifier = MuleAccountClassifier()
        classifier.fit(accounts, labels)
        classifier.save()

        # Load in new instance
        new_classifier = MuleAccountClassifier()
        loaded = new_classifier.load()

        assert loaded
        assert new_classifier.is_fitted

        # Compare predictions
        test_account = accounts[0]
        pred1 = classifier.predict(test_account)
        pred2 = new_classifier.predict(test_account)

        assert pred1.classification == pred2.classification