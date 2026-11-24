"""Synthetic fraud data generator for FinShield."""
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from faker import Faker
import numpy as np

from src.common.models import (
    Transaction, TransactionChannel, TransactionType,
    FMSAlert, TMSAlert, GovernmentWatchlistEntry
)
from src.common.config import get_config

fake = Faker('en_IN')
Faker.seed(42)
random.seed(42)
np.random.seed(42)


class SyntheticDataGenerator:
    """Generates realistic synthetic banking transaction data with embedded fraud patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()._config
        self.synth_config = self.config.get("ingestion", {}).get("synthetic_data", {})
        self.num_accounts = self.synth_config.get("num_accounts", 1000)
        self.num_transactions = self.synth_config.get("num_transactions", 50000)
        self.fraud_ratio = self.synth_config.get("fraud_ratio", 0.15)

        # Account pools
        self.legitimate_accounts: List[str] = []
        self.mule_accounts: List[str] = []
        self.suspicious_accounts: List[str] = []
        self.kingpin_accounts: List[str] = []
        self.all_accounts: List[str] = []

        # Account profiles (for test compatibility)
        self.account_profiles: Dict[str, Dict[str, Any]] = {}

        # Device/IP pools for shared entity detection
        self.devices: List[str] = [f"DEV_{uuid.uuid4().hex[:8]}" for _ in range(200)]
        self.ips: List[str] = [f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(150)]
        self.phones: List[str] = [f"+91{random.randint(7000000000, 9999999999)}" for _ in range(300)]

        # Fraud rings (groups of mule accounts controlled by same entity)
        self.fraud_rings: List[List[str]] = []

    def generate_accounts(self) -> Dict[str, List[str]]:
        """Generate account IDs with fraud labels."""
        num_fraud = int(self.num_accounts * self.fraud_ratio)
        num_mule = int(num_fraud * 0.6)  # 60% mule
        num_suspicious = num_fraud - num_mule
        num_legitimate = self.num_accounts - num_fraud

        self.legitimate_accounts = [f"ACC_LEG_{uuid.uuid4().hex[:8]}" for _ in range(num_legitimate)]
        self.mule_accounts = [f"ACC_MULE_{uuid.uuid4().hex[:8]}" for _ in range(num_mule)]
        self.suspicious_accounts = [f"ACC_SUS_{uuid.uuid4().hex[:8]}" for _ in range(num_suspicious)]
        self.all_accounts = self.legitimate_accounts + self.mule_accounts + self.suspicious_accounts

        # Create account profiles for each account (test compatibility)
        for acc in self.legitimate_accounts:
            self.account_profiles[acc] = {"type": "legitimate", "risk_level": "low"}
        for acc in self.mule_accounts:
            self.account_profiles[acc] = {"type": "mule", "risk_level": "high"}
        for acc in self.suspicious_accounts:
            self.account_profiles[acc] = {"type": "suspicious", "risk_level": "medium"}

        # Create fraud rings (5-15 mules per ring, 1 kingpin)
        mule_copy = self.mule_accounts.copy()
        random.shuffle(mule_copy)

        ring_size = random.randint(5, 15)
        for i in range(0, len(mule_copy), ring_size):
            ring = mule_copy[i:i+ring_size]
            if len(ring) >= 3:
                kingpin = f"ACC_KING_{uuid.uuid4().hex[:8]}"
                self.kingpin_accounts.append(kingpin)
                self.fraud_rings.append(ring + [kingpin])
                self.all_accounts.append(kingpin)
                self.account_profiles[kingpin] = {"type": "kingpin", "risk_level": "critical"}

        return {
            "legitimate": self.legitimate_accounts,
            "mule": self.mule_accounts,
            "suspicious": self.suspicious_accounts,
            "kingpin": self.kingpin_accounts,
            "fraud_rings": self.fraud_rings
        }

    def generate_transactions(self) -> List[Transaction]:
        """Generate transactions with embedded fraud patterns."""
        if not self.all_accounts:
            self.generate_accounts()

        transactions = []
        base_date = datetime.now() - timedelta(days=90)

        # Generate legitimate transactions
        for account in self.legitimate_accounts:
            txns = self._generate_legitimate_pattern(account, base_date)
            transactions.extend(txns)

        # Generate mule account transactions (structured, layered, circular)
        for ring in self.fraud_rings:
            mules = ring[:-1]
            kingpin = ring[-1]
            txns = self._generate_mule_ring_pattern(mules, kingpin, base_date)
            transactions.extend(txns)

        # Generate suspicious account transactions (anomalous but not clear mule)
        for account in self.suspicious_accounts:
            txns = self._generate_suspicious_pattern(account, base_date)
            transactions.extend(txns)

        # Shuffle and limit
        random.shuffle(transactions)
        return transactions[:self.num_transactions]

    def _generate_legitimate_pattern(self, account_id: str, base_date: datetime) -> List[Transaction]:
        """Generate normal user transaction pattern."""
        transactions = []
        num_txns = random.randint(50, 300)

        # User profile
        preferred_hours = [random.randint(8, 22) for _ in range(3)]
        preferred_channels = random.sample(list(TransactionChannel), k=random.randint(2, 4))
        avg_amount = random.uniform(500, 50000)

        device = random.choice(self.devices)
        ip = random.choice(self.ips)
        phone = random.choice(self.phones)

        beneficiaries = [f"BEN_{uuid.uuid4().hex[:6]}" for _ in range(random.randint(3, 10))]

        for _ in range(num_txns):
            day_offset = random.randint(0, 89)
            hour = random.choice(preferred_hours) + random.randint(-2, 2)
            hour = max(0, min(23, hour))
            minute = random.randint(0, 59)

            timestamp = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

            channel = random.choice(preferred_channels)
            txn_type = random.choices(
                [TransactionType.CREDIT, TransactionType.DEBIT],
                weights=[0.3, 0.7]
            )[0]

            # Amount with some variance
            amount = max(10, np.random.lognormal(np.log(avg_amount), 0.8))

            beneficiary = random.choice(beneficiaries) if txn_type == TransactionType.DEBIT else None

            txn = Transaction(
                account_id=account_id,
                amount=round(amount, 2),
                channel=channel,
                txn_type=txn_type,
                timestamp=timestamp,
                beneficiary_account_id=beneficiary,
                beneficiary_name=fake.name() if beneficiary else None,
                device_id=device if random.random() > 0.1 else None,
                ip_address=ip if random.random() > 0.1 else None,
                phone_number=phone if random.random() > 0.2 else None,
                location=fake.city() if random.random() > 0.3 else None
            )
            transactions.append(txn)

        return transactions

    def _generate_mule_ring_pattern(self, mules: List[str], kingpin: str, base_date: datetime) -> List[Transaction]:
        """Generate mule ring pattern: structuring, smurfing, circular flows."""
        transactions = []

        # Shared device/IP for mule farm detection
        shared_device = random.choice(self.devices)
        shared_ip = random.choice(self.ips)
        shared_phone = random.choice(self.phones)

        # Phase 1: Inbound funds to mules (structuring - many small deposits)
        for mule in mules:
            num_deposits = random.randint(10, 30)
            for i in range(num_deposits):
                day_offset = random.randint(0, 30)
                hour = random.randint(0, 23)  # Mules operate at odd hours
                timestamp = base_date + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))

                # Structuring: amounts just under reporting threshold (e.g., ₹50,000)
                amount = random.uniform(45000, 49999)

                txn = Transaction(
                    account_id=mule,
                    amount=round(amount, 2),
                    channel=random.choice([TransactionChannel.UPI, TransactionChannel.IMPS, TransactionChannel.NEFT]),
                    txn_type=TransactionType.CREDIT,
                    timestamp=timestamp,
                    beneficiary_account_id=f"EXT_{uuid.uuid4().hex[:8]}",
                    device_id=shared_device,
                    ip_address=shared_ip,
                    phone_number=shared_phone
                )
                transactions.append(txn)

        # Phase 2: Mule to kingpin transfers (layering)
        for mule in mules:
            num_transfers = random.randint(3, 8)
            for _ in range(num_transfers):
                day_offset = random.randint(10, 60)
                hour = random.randint(0, 23)
                timestamp = base_date + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))

                # Transfer most funds to kingpin
                amount = random.uniform(40000, 49999)

                txn = Transaction(
                    account_id=mule,
                    amount=round(amount, 2),
                    channel=random.choice([TransactionChannel.UPI, TransactionChannel.IMPS, TransactionChannel.NEFT]),
                    txn_type=TransactionType.DEBIT,
                    timestamp=timestamp,
                    beneficiary_account_id=kingpin,
                    device_id=shared_device,
                    ip_address=shared_ip
                )
                transactions.append(txn)

                # Corresponding credit to kingpin
                txn_kingpin = Transaction(
                    account_id=kingpin,
                    amount=round(amount, 2),
                    channel=TransactionChannel.UPI,
                    txn_type=TransactionType.CREDIT,
                    timestamp=timestamp,
                    beneficiary_account_id=mule,
                    device_id=shared_device,
                    ip_address=shared_ip
                )
                transactions.append(txn_kingpin)

        # Phase 3: Circular flows (money laundering)
        for _ in range(random.randint(5, 15)):
            mule_a = random.choice(mules)
            mule_b = random.choice(mules)
            if mule_a != mule_b:
                day_offset = random.randint(20, 70)
                timestamp = base_date + timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
                amount = random.uniform(10000, 30000)

                # A -> B
                transactions.append(Transaction(
                    account_id=mule_a, amount=round(amount, 2),
                    channel=TransactionChannel.UPI, txn_type=TransactionType.DEBIT,
                    timestamp=timestamp, beneficiary_account_id=mule_b,
                    device_id=shared_device, ip_address=shared_ip
                ))
                # B -> A (circular)
                transactions.append(Transaction(
                    account_id=mule_b, amount=round(amount * 0.95, 2),
                    channel=TransactionChannel.UPI, txn_type=TransactionType.DEBIT,
                    timestamp=timestamp + timedelta(minutes=random.randint(5, 60)),
                    beneficiary_account_id=mule_a, device_id=shared_device, ip_address=shared_ip
                ))

        # Phase 4: Kingpin cash-out (large transfers to external)
        for _ in range(random.randint(5, 15)):
            day_offset = random.randint(40, 85)
            timestamp = base_date + timedelta(days=day_offset, hours=random.randint(9, 18), minutes=random.randint(0, 59))
            amount = random.uniform(100000, 500000)

            transactions.append(Transaction(
                account_id=kingpin, amount=round(amount, 2),
                channel=random.choice([TransactionChannel.NEFT, TransactionChannel.RTGS]),
                txn_type=TransactionType.DEBIT, timestamp=timestamp,
                beneficiary_account_id=f"EXT_CASHOUT_{uuid.uuid4().hex[:8]}",
                device_id=shared_device, ip_address=shared_ip
            ))

        return transactions

    def _generate_suspicious_pattern(self, account_id: str, base_date: datetime) -> List[Transaction]:
        """Generate suspicious but not clearly mule pattern."""
        transactions = []
        num_txns = random.randint(30, 150)

        # Dormant then sudden activity
        dormant_days = random.randint(60, 85)
        active_start = base_date + timedelta(days=dormant_days)

        device = random.choice(self.devices)
        ip = random.choice(self.ips)

        for i in range(num_txns):
            day_offset = dormant_days + random.randint(0, 5)
            hour = random.randint(0, 23)
            timestamp = base_date + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))

            # High velocity, round amounts
            amount = round(random.uniform(10000, 99000) / 1000) * 1000

            txn = Transaction(
                account_id=account_id,
                amount=amount,
                channel=random.choice([TransactionChannel.UPI, TransactionChannel.IMPS, TransactionChannel.NEFT]),
                txn_type=random.choice([TransactionType.CREDIT, TransactionType.DEBIT]),
                timestamp=timestamp,
                beneficiary_account_id=f"BEN_{uuid.uuid4().hex[:6]}",
                device_id=device,
                ip_address=ip
            )
            transactions.append(txn)

        return transactions

    def generate_fms_alerts(self, accounts: List[str]) -> List[FMSAlert]:
        """Generate synthetic FMS alerts."""
        alerts = []
        alert_types = ["HIGH_VELOCITY", "STRUCTURING", "UNUSUAL_CHANNEL", "GEO_ANOMALY", "DEVICE_ANOMALY"]
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        for account in random.sample(accounts, min(200, len(accounts))):
            num_alerts = random.randint(1, 5)
            for _ in range(num_alerts):
                alerts.append(FMSAlert(
                    account_id=account,
                    alert_type=random.choice(alert_types),
                    severity=random.choice(severities),
                    description=f"Automated alert: {random.choice(alert_types)} detected",
                    source_system="NICE_Actimize",
                    timestamp=datetime.now() - timedelta(days=random.randint(0, 30))
                ))
        return alerts

    def generate_tms_alerts(self, accounts: List[str]) -> List[TMSAlert]:
        """Generate synthetic TMS alerts."""
        alerts = []
        rule_names = ["VELOCITY_CHECK", "AMOUNT_THRESHOLD", "COUNTERPARTY_RISK", "DORMANCY_BREAK", "ROUND_AMOUNT"]

        for account in random.sample(accounts, min(150, len(accounts))):
            num_alerts = random.randint(1, 3)
            for _ in range(num_alerts):
                alerts.append(TMSAlert(
                    account_id=account,
                    rule_id=f"RULE_{random.randint(100, 999)}",
                    rule_name=random.choice(rule_names),
                    risk_score=random.uniform(30, 95),
                    timestamp=datetime.now() - timedelta(days=random.randint(0, 30)),
                    transaction_ids=[str(uuid.uuid4()) for _ in range(random.randint(1, 5))]
                ))
        return alerts

    def generate_watchlist_entries(self) -> List[GovernmentWatchlistEntry]:
        """Generate synthetic government watchlist entries."""
        entries = []
        sources = ["RBI", "FIU-IND", "NPCI", "MHA"]
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # Add some known mule/kingpin accounts to watchlist
        watchlisted = random.sample(self.mule_accounts + self.kingpin_accounts, min(50, len(self.mule_accounts + self.kingpin_accounts)))

        for account in watchlisted:
            entries.append(GovernmentWatchlistEntry(
                source=random.choice(sources),
                account_id=account,
                entity_type="account",
                identifiers={"account_number": account},
                listed_date=datetime.now() - timedelta(days=random.randint(1, 180)),
                reason="Suspected money mule account",
                severity=random.choice(severities)
            ))

        # Add some random entries
        for _ in range(20):
            entries.append(GovernmentWatchlistEntry(
                source=random.choice(sources),
                account_id=f"ACC_WL_{uuid.uuid4().hex[:8]}",
                entity_type="account",
                identifiers={"account_number": f"ACC_WL_{uuid.uuid4().hex[:8]}"},
                listed_date=datetime.now() - timedelta(days=random.randint(1, 365)),
                reason="Fraud reported",
                severity=random.choice(severities)
            ))

        return entries


def generate_all_data() -> Dict[str, Any]:
    """Generate complete synthetic dataset."""
    generator = SyntheticDataGenerator()

    print("Generating accounts...")
    accounts = generator.generate_accounts()

    print("Generating transactions...")
    transactions = generator.generate_transactions()

    print("Generating FMS alerts...")
    fms_alerts = generator.generate_fms_alerts(generator.all_accounts)

    print("Generating TMS alerts...")
    tms_alerts = generator.generate_tms_alerts(generator.all_accounts)

    print("Generating watchlist entries...")
    watchlist = generator.generate_watchlist_entries()

    print(f"Generated: {len(transactions)} transactions, {len(fms_alerts)} FMS alerts, {len(tms_alerts)} TMS alerts, {len(watchlist)} watchlist entries")

    return {
        "accounts": accounts,
        "transactions": transactions,
        "fms_alerts": fms_alerts,
        "tms_alerts": tms_alerts,
        "watchlist": watchlist
    }


if __name__ == "__main__":
    data = generate_all_data()
    print("Data generation complete!")