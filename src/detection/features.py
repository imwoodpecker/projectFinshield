"""Feature extraction for FinShield detection engine."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from src.common.models import Transaction, TransactionChannel, TransactionType
from src.ingestion.database import get_db
from src.common.config import get_config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from transaction data for ML models."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.baseline_window_days = self.config.get("detection.baseline.window_days", 90)

    def extract_transaction_features(self, account_id: str, days: int = 90) -> Dict[str, float]:
        """Extract comprehensive features for an account."""
        df = self.db.get_account_transactions(account_id, days)

        if df.empty:
            return self._empty_features()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        features = {}

        # Basic transaction features
        features.update(self._basic_features(df))
        features.update(self._amount_features(df))
        features.update(self._velocity_features(df))
        features.update(self._temporal_features(df))
        features.update(self._channel_features(df))
        features.update(self._beneficiary_features(df))
        features.update(self._device_features(df))
        features.update(self._dormancy_features(df))
        features.update(self._structuring_features(df))
        features.update(self._round_amount_features(df))

        return features

    def _empty_features(self) -> Dict[str, float]:
        return {
            'txn_count': 0, 'total_amount': 0, 'avg_amount': 0, 'std_amount': 0,
            'min_amount': 0, 'max_amount': 0, 'credit_count': 0, 'debit_count': 0,
            'credit_amount': 0, 'debit_amount': 0, 'net_flow': 0,
            'txn_per_day': 0, 'avg_txn_per_hour': 0, 'max_txn_per_hour': 0,
            'unique_channels': 0, 'channel_entropy': 0,
            'unique_beneficiaries': 0, 'beneficiary_concentration': 0,
            'unique_devices': 0, 'unique_ips': 0,
            'dormancy_max_days': 0, 'dormancy_avg_days': 0,
            'structuring_score': 0, 'round_amount_ratio': 0,
            'night_txn_ratio': 0, 'weekend_txn_ratio': 0,
            'amount_cv': 0, 'time_gap_avg': 0, 'time_gap_std': 0
        }

    def _basic_features(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            'txn_count': len(df),
            'total_amount': df['amount'].sum(),
            'avg_amount': df['amount'].mean(),
            'std_amount': df['amount'].std() if len(df) > 1 else 0,
            'min_amount': df['amount'].min(),
            'max_amount': df['amount'].max(),
            'credit_count': len(df[df['txn_type'] == 'CREDIT']),
            'debit_count': len(df[df['txn_type'] == 'DEBIT']),
            'credit_amount': df[df['txn_type'] == 'CREDIT']['amount'].sum(),
            'debit_amount': df[df['txn_type'] == 'DEBIT']['amount'].sum(),
            'net_flow': df[df['txn_type'] == 'CREDIT']['amount'].sum() - df[df['txn_type'] == 'DEBIT']['amount'].sum(),
        }

    def _amount_features(self, df: pd.DataFrame) -> Dict[str, float]:
        amounts = df['amount']
        return {
            'amount_cv': amounts.std() / amounts.mean() if amounts.mean() > 0 else 0,
            'amount_skew': amounts.skew() if len(df) > 2 else 0,
            'amount_kurtosis': amounts.kurtosis() if len(df) > 3 else 0,
            'amount_q25': amounts.quantile(0.25),
            'amount_q75': amounts.quantile(0.75),
            'amount_iqr': amounts.quantile(0.75) - amounts.quantile(0.25),
        }

    def _velocity_features(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()

        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour').size()

        # Velocity score: normalized transaction frequency
        txn_per_day = daily_counts.mean() if len(daily_counts) > 0 else 0
        velocity_score = min(txn_per_day / 50, 1.0)  # Normalize: 50 txns/day = max velocity

        return {
            'txn_per_day': txn_per_day,
            'txn_per_day_std': daily_counts.std() if len(daily_counts) > 1 else 0,
            'avg_txn_per_hour': hourly_counts.mean() if len(hourly_counts) > 0 else 0,
            'max_txn_per_hour': hourly_counts.max() if len(hourly_counts) > 0 else 0,
            'active_days': len(daily_counts),
            'active_hours': len(hourly_counts),
            'velocity_score': velocity_score,
        }

    def _temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df = df.sort_values('timestamp')
        df['time_gap'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes

        night_txns = len(df[(df['hour'] >= 22) | (df['hour'] <= 5)])
        weekend_txns = len(df[df['dayofweek'] >= 5])

        return {
            'night_txn_ratio': night_txns / len(df) if len(df) > 0 else 0,
            'weekend_txn_ratio': weekend_txns / len(df) if len(df) > 0 else 0,
            'preferred_hour': df['hour'].mode().iloc[0] if len(df) > 0 else 12,
            'hour_entropy': self._entropy(df['hour'].value_counts()),
            'day_entropy': self._entropy(df['dayofweek'].value_counts()),
            'time_gap_avg': df['time_gap'].mean() if len(df) > 1 else 0,
            'time_gap_std': df['time_gap'].std() if len(df) > 2 else 0,
            'time_gap_min': df['time_gap'].min() if len(df) > 1 else 0,
            'burst_score': (df['time_gap'] < 5).sum() / len(df) if len(df) > 1 else 0,  # txns within 5 min
        }

    def _channel_features(self, df: pd.DataFrame) -> Dict[str, float]:
        channel_counts = df['channel'].value_counts()
        return {
            'unique_channels': len(channel_counts),
            'channel_entropy': self._entropy(channel_counts),
            'upi_ratio': channel_counts.get('UPI', 0) / len(df),
            'neft_ratio': channel_counts.get('NEFT', 0) / len(df),
            'imps_ratio': channel_counts.get('IMPS', 0) / len(df),
            'atm_ratio': channel_counts.get('ATM', 0) / len(df),
            'card_ratio': channel_counts.get('CARD', 0) / len(df),
        }

    def _beneficiary_features(self, df: pd.DataFrame) -> Dict[str, float]:
        debit_df = df[df['txn_type'] == 'DEBIT']
        if len(debit_df) == 0 or debit_df['beneficiary_account_id'].isna().all():
            return {'unique_beneficiaries': 0, 'beneficiary_concentration': 0, 'new_beneficiary_ratio': 0}

        ben_counts = debit_df['beneficiary_account_id'].dropna().value_counts()
        return {
            'unique_beneficiaries': len(ben_counts),
            'beneficiary_concentration': ben_counts.iloc[0] / len(debit_df) if len(ben_counts) > 0 else 0,
            'beneficiary_entropy': self._entropy(ben_counts),
            'new_beneficiary_ratio': (ben_counts == 1).sum() / len(ben_counts) if len(ben_counts) > 0 else 0,
        }

    def _device_features(self, df: pd.DataFrame) -> Dict[str, float]:
        device_counts = df['device_id'].dropna().value_counts()
        ip_counts = df['ip_address'].dropna().value_counts()
        return {
            'unique_devices': len(device_counts),
            'device_entropy': self._entropy(device_counts),
            'primary_device_ratio': device_counts.iloc[0] / len(df) if len(device_counts) > 0 else 0,
            'unique_ips': len(ip_counts),
            'ip_entropy': self._entropy(ip_counts),
            'primary_ip_ratio': ip_counts.iloc[0] / len(df) if len(ip_counts) > 0 else 0,
        }

    def _dormancy_features(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        daily_dates = df['date'].unique()
        daily_dates = np.sort(daily_dates)

        if len(daily_dates) < 2:
            return {
                'dormancy_max_days': 0, 'dormancy_avg_days': 0, 'dormancy_std_days': 0,
                'current_dormancy_days': 0, 'longest_active_streak': 0,
                'days_since_last_txn': 0, 'dormancy_break_flag': 0
            }

        gaps = [(daily_dates[i+1] - daily_dates[i]).days for i in range(len(daily_dates)-1)]
        current_dormancy = (datetime.now().date() - daily_dates[-1]).days

        # Days since last transaction
        days_since_last_txn = current_dormancy

        # Dormancy break flag: 1 if there was a long dormancy followed by activity
        dormancy_break_flag = 1 if any(g > 30 for g in gaps) and current_dormancy < 7 else 0

        return {
            'dormancy_max_days': max(gaps) if gaps else 0,
            'dormancy_avg_days': np.mean(gaps) if gaps else 0,
            'dormancy_std_days': np.std(gaps) if len(gaps) > 1 else 0,
            'current_dormancy_days': current_dormancy,
            'longest_active_streak': self._longest_streak(gaps),
            'days_since_last_txn': days_since_last_txn,
            'dormancy_break_flag': dormancy_break_flag,
        }

    def _structuring_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect structuring (amounts just below reporting thresholds)."""
        # Common thresholds in India: 50k, 10L, 50L, 1Cr
        thresholds = [50000, 1000000, 5000000, 10000000]
        structuring_count = 0

        for _, row in df.iterrows():
            amount = row['amount']
            for threshold in thresholds:
                # Within 1% below threshold
                if threshold * 0.99 <= amount < threshold:
                    structuring_count += 1
                    break

        return {
            'structuring_score': structuring_count / len(df) if len(df) > 0 else 0,
            'structuring_count': structuring_count,
        }

    def _round_amount_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect suspiciously round amounts."""
        amounts = df['amount']
        round_1000 = ((amounts % 1000) == 0).sum()
        round_10000 = ((amounts % 10000) == 0).sum()
        round_100000 = ((amounts % 100000) == 0).sum()

        return {
            'round_amount_ratio': round_1000 / len(df) if len(df) > 0 else 0,
            'round_10k_ratio': round_10000 / len(df) if len(df) > 0 else 0,
            'round_100k_ratio': round_100000 / len(df) if len(df) > 0 else 0,
        }

    def _entropy(self, counts: pd.Series) -> float:
        """Calculate Shannon entropy."""
        if len(counts) == 0:
            return 0
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def _longest_streak(self, gaps: List[int]) -> int:
        """Calculate longest streak of consecutive active days."""
        streak = 1
        max_streak = 1
        for gap in gaps:
            if gap == 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        return max_streak

    def extract_tsfresh_features(self, account_id: str, days: int = 90) -> pd.DataFrame:
        """Extract tsfresh time-series features."""
        try:
            from tsfresh import extract_features
            from tsfresh.utilities.dataframe_functions import impute
        except ImportError:
            logger.warning("tsfresh not available")
            return pd.DataFrame()

        df = self.db.get_account_transactions(account_id, days)
        if df.empty:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['id'] = account_id
        df['time'] = range(len(df))
        df['value'] = df['amount']

        # Extract features
        features = extract_features(
            df[['id', 'time', 'value']],
            column_id='id',
            column_sort='time',
            column_value='value',
            n_jobs=self.config.get("detection.tsfresh.n_jobs", 4),
            disable_progressbar=self.config.get("detection.tsfresh.disable_progressbar", True)
        )

        impute(features)
        return features

    def get_account_baseline(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get stored 90-day baseline profile."""
        return self.db.get_account_profile(account_id)

    def compute_baseline_deviation(self, account_id: str, current_features: Dict[str, float]) -> float:
        """Compute deviation from historical baseline."""
        baseline = self.get_account_baseline(account_id)
        if not baseline:
            return 0.5  # No baseline, moderate deviation

        # Key metrics to compare
        key_metrics = [
            'avg_txn_amount', 'txn_count', 'avg_daily_txn_count',
            'velocity_score'
        ]

        deviations = []
        for metric in key_metrics:
            baseline_val = baseline.get(metric, 0)
            current_val = current_features.get(metric, 0)
            if baseline_val > 0:
                dev = abs(current_val - baseline_val) / baseline_val
                deviations.append(min(dev, 5.0))  # Cap at 5x

        return np.mean(deviations) if deviations else 0.5


def build_feature_matrix(accounts: List[str], days: int = 90) -> pd.DataFrame:
    """Build feature matrix for multiple accounts."""
    extractor = FeatureExtractor()
    features_list = []

    for account_id in accounts:
        features = extractor.extract_transaction_features(account_id, days)
        features['account_id'] = account_id
        features_list.append(features)

    return pd.DataFrame(features_list)


# Add extract_all_features method to FeatureExtractor class for test compatibility
def extract_all_features(self, account_list: List[str], transactions: List[Any] = None) -> pd.DataFrame:
    """Extract features for multiple accounts - test compatibility method.

    If transactions are provided, use them directly. Otherwise, query from database.
    """
    if transactions is not None:
        # Use provided transactions - create a temporary extractor that uses these
        from src.detection.features import FeatureExtractor
        extractor = FeatureExtractor()
        # We need to temporarily override the database to use our transactions
        # For simplicity, just use build_feature_matrix which queries DB
        # But since test passes transactions, we should use them
        import pandas as pd
        from src.common.models import Transaction

        # Convert transactions to DataFrame
        txn_data = []
        for txn in transactions:
            if hasattr(txn, 'account_id'):
                # It's a Transaction object
                txn_data.append({
                    'transaction_id': txn.transaction_id,
                    'account_id': txn.account_id,
                    'amount': txn.amount,
                    'currency': txn.currency,
                    'channel': txn.channel.value if hasattr(txn.channel, 'value') else str(txn.channel),
                    'txn_type': txn.txn_type.value if hasattr(txn.txn_type, 'value') else str(txn.txn_type),
                    'timestamp': txn.timestamp,
                    'beneficiary_account_id': txn.beneficiary_account_id,
                    'beneficiary_name': txn.beneficiary_name,
                    'device_id': txn.device_id,
                    'ip_address': txn.ip_address,
                    'phone_number': txn.phone_number,
                    'location': txn.location,
                    'metadata': txn.metadata if hasattr(txn, 'metadata') else {}
                })
            else:
                # It's already a dict
                txn_data.append(txn)

        if not txn_data:
            return pd.DataFrame()

        df = pd.DataFrame(txn_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        features_list = []
        for account_id in account_list:
            acc_df = df[df['account_id'] == account_id].copy()
            if acc_df.empty:
                features = extractor._empty_features()
            else:
                acc_df = acc_df.sort_values('timestamp')
                features = {}
                features.update(extractor._basic_features(acc_df))
                features.update(extractor._amount_features(acc_df))
                features.update(extractor._velocity_features(acc_df))
                features.update(extractor._temporal_features(acc_df))
                features.update(extractor._channel_features(acc_df))
                features.update(extractor._beneficiary_features(acc_df))
                features.update(extractor._device_features(acc_df))
                features.update(extractor._dormancy_features(acc_df))
                features.update(extractor._structuring_features(acc_df))
                features.update(extractor._round_amount_features(acc_df))
            features['account_id'] = account_id
            features_list.append(features)

        return pd.DataFrame(features_list)

    # Fall back to database query
    return build_feature_matrix(account_list)

FeatureExtractor.extract_all_features = extract_all_features