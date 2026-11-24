"""Unit tests for ingestion layer."""
import pytest
import pandas as pd
from datetime import datetime
from src.ingestion.database import DuckDBManager
from src.ingestion.synthetic_data import SyntheticDataGenerator
from src.common.models import (
    Transaction, TransactionChannel, TransactionType,
    FMSAlert, TMSAlert, GovernmentWatchlistEntry
)


class TestDuckDBManager:
    """Tests for DuckDB database manager."""

    @pytest.fixture
    def db(self):
        """Create test database."""
        db = DuckDBManager(":memory:")
        yield db
        db.close()

    def test_insert_transaction(self, db):
        """Test single transaction insertion."""
        txn = Transaction(
            transaction_id="TXN001",
            account_id="ACC001",
            amount=10000.0,
            channel=TransactionChannel.IMPS,
            txn_type=TransactionType.CREDIT,
            timestamp=datetime.utcnow(),
            beneficiary_account_id="ACC002"
        )
        db.insert_transaction(txn)

        result = db.get_account_transactions("ACC001", days=30)
        assert len(result) == 1
        assert result.iloc[0]['transaction_id'] == "TXN001"

    def test_insert_batch_transactions(self, db):
        """Test batch transaction insertion."""
        txns = [
            Transaction(
                transaction_id=f"TXN{i:03d}",
                account_id="ACC001",
                amount=1000.0 * i,
                channel=TransactionChannel.UPI,
                txn_type=TransactionType.DEBIT,
                timestamp=datetime.utcnow()
            )
            for i in range(10)
        ]
        db.insert_transactions_batch(txns)

        result = db.get_account_transactions("ACC001", days=30)
        assert len(result) == 10

    def test_fms_alert(self, db):
        """Test FMS alert insertion."""
        alert = FMSAlert(
            alert_id="FMS001",
            account_id="ACC001",
            alert_type="HIGH_VALUE",
            severity="HIGH",
            description="Test alert",
            source_system="TEST_SYSTEM",
            timestamp=datetime.utcnow()
        )
        db.insert_fms_alert(alert)

        conn = db._get_connection()
        result = conn.execute("SELECT * FROM fms_alerts WHERE alert_id = 'FMS001'").fetchone()
        assert result is not None
        assert result[2] == "HIGH_VALUE"

    def test_watchlist_entry(self, db):
        """Test watchlist entry insertion."""
        entry = GovernmentWatchlistEntry(
            watchlist_id="WL001",
            source="TEST_SOURCE",
            account_id="ACC001",
            entity_name="Test Entity",
            entity_type="INDIVIDUAL",
            identifiers={"pan": "ABCDE1234F"},
            listed_date=datetime.utcnow(),
            reason="Test reason",
            severity="HIGH"
        )
        db.insert_watchlist_entry(entry)

        matches = db.is_account_watchlisted("ACC001")
        assert len(matches) == 1
        assert matches[0]['entity_name'] == "Test Entity"


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation."""

    @pytest.fixture
    def generator(self):
        """Create test generator with small config."""
        config = {
            "ingestion": {
                "synthetic_data": {
                    "num_accounts": 100,
                    "num_transactions": 5000,
                    "fraud_ratio": 0.2
                }
            }
        }
        return SyntheticDataGenerator(config=config)

    def test_generate_accounts(self, generator):
        """Test account generation creates all types."""
        accounts = generator.generate_accounts()

        assert 'legitimate' in accounts
        assert 'mule' in accounts
        assert 'suspicious' in accounts
        assert 'kingpin' in accounts
        assert 'fraud_rings' in accounts

        assert len(accounts['legitimate']) > 0
        assert len(accounts['mule']) > 0
        assert len(accounts['suspicious']) > 0

        # Check prefixes
        for acc in accounts['legitimate']:
            assert acc.startswith("ACC_LEG_")
        for acc in accounts['mule']:
            assert acc.startswith("ACC_MULE_")
        for acc in accounts['suspicious']:
            assert acc.startswith("ACC_SUS_")
        for acc in accounts['kingpin']:
            assert acc.startswith("ACC_KING_")

    def test_generate_transactions(self, generator):
        """Test transaction generation."""
        generator.generate_accounts()
        transactions = generator.generate_transactions()

        assert len(transactions) > 0
        assert len(transactions) <= 5000  # Limited by config
        for txn in transactions:
            assert txn.amount > 0
            assert txn.account_id in generator.all_accounts

    def test_generate_fms_alerts(self, generator):
        """Test FMS alert generation."""
        generator.generate_accounts()
        alerts = generator.generate_fms_alerts(generator.all_accounts)

        assert len(alerts) > 0
        for alert in alerts:
            assert alert.account_id in generator.all_accounts
            assert alert.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_generate_tms_alerts(self, generator):
        """Test TMS alert generation."""
        generator.generate_accounts()
        alerts = generator.generate_tms_alerts(generator.all_accounts)

        assert len(alerts) > 0
        for alert in alerts:
            assert alert.account_id in generator.all_accounts
            assert 0 <= alert.risk_score <= 100

    def test_generate_watchlist_entries(self, generator):
        """Test watchlist entry generation."""
        generator.generate_accounts()
        entries = generator.generate_watchlist_entries()

        assert len(entries) > 0
        for entry in entries:
            assert entry.source in ["RBI", "FIU-IND", "NPCI", "MHA"]
            assert entry.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]