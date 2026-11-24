"""XGBoost classifier for mule account detection with SHAP explainability."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

from src.common.models import RiskAssessment, RiskTier
from src.ingestion.database import get_db
from src.common.config import get_config
from src.detection.anomaly import AnomalyDetector
from src.graph.analysis import GraphAnalyzer

logger = logging.getLogger(__name__)


class MuleAccountClassifier:
    """Supervised XGBoost classifier for mule account detection."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.model_dir = Path("models/classifier")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Anomaly detector and graph analyzer for feature enrichment
        self.anomaly_detector = AnomalyDetector()
        self.graph_analyzer = GraphAnalyzer()

    def prepare_features(self, accounts: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Build comprehensive feature matrix combining all signals."""
        from src.detection.features import FeatureExtractor, build_feature_matrix

        logger.info(f"Building features for {len(accounts)} accounts...")

        # Base transaction features
        feature_df = build_feature_matrix(accounts)

        if feature_df.empty:
            return pd.DataFrame(), []

        # Get anomaly scores
        anomaly_scores = []
        for acc in accounts:
            score = self.db.get_latest_risk_assessment(acc)
            if score:
                anomaly_scores.append(score)
            else:
                anomaly_scores.append({})

        # Add anomaly features
        for i, acc in enumerate(accounts):
            if i < len(anomaly_scores) and anomaly_scores[i]:
                a = anomaly_scores[i]
                feature_df.loc[feature_df['account_id'] == acc, 'anomaly_iforest'] = a.get('isolation_forest_score', 0)
                feature_df.loc[feature_df['account_id'] == acc, 'anomaly_ecod'] = a.get('ecod_score', 0)
                feature_df.loc[feature_df['account_id'] == acc, 'anomaly_loda'] = a.get('loda_score', 0)
                feature_df.loc[feature_df['account_id'] == acc, 'anomaly_ensemble'] = a.get('ensemble_score', 0)
            else:
                # Ensure anomaly_ensemble column exists
                feature_df.loc[feature_df['account_id'] == acc, 'anomaly_ensemble'] = 0

        # Get graph metrics
        for acc in accounts:
            graph_metrics = self.graph_analyzer.get_account_graph_metrics(acc)
            if graph_metrics:
                for key, val in graph_metrics.items():
                    feature_df.loc[feature_df['account_id'] == acc, f'graph_{key}'] = val

        # Get watchlist matches
        for acc in accounts:
            matches = self.db.is_account_watchlisted(acc)
            feature_df.loc[feature_df['account_id'] == acc, 'watchlist_match_count'] = len(matches)
            feature_df.loc[feature_df['account_id'] == acc, 'watchlist_max_severity'] = max(
                [1 if m['severity'] == 'LOW' else 2 if m['severity'] == 'MEDIUM' else 3 if m['severity'] == 'HIGH' else 4
                 for m in matches], default=0
            )

        # Fill NaN
        feature_cols = [c for c in feature_df.columns if c != 'account_id']
        feature_df[feature_cols] = feature_df[feature_cols].fillna(0)

        self.feature_names = feature_cols
        return feature_df, feature_cols

    def create_labels(self, accounts: List[str]) -> Dict[str, str]:
        """Create labels for training. In production, these come from historical investigation results.
        For synthetic data, we use the known account types."""
        labels = {}
        for acc in accounts:
            if 'MULE' in acc:
                labels[acc] = 'MULE'
            elif 'KING' in acc:
                labels[acc] = 'MULE'  # Kingpins are also mule accounts
            elif 'SUS' in acc:
                labels[acc] = 'SUSPICIOUS'
            else:
                labels[acc] = 'LEGITIMATE'
        return labels

    def fit(self, accounts: List[str], labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Train the XGBoost classifier."""
        logger.info("Training XGBoost classifier...")

        # Prepare features
        feature_df, feature_cols = self.prepare_features(accounts)
        if feature_df.empty:
            raise ValueError("No feature data available for training")

        # Create labels if not provided
        if labels is None:
            labels = self.create_labels(accounts)

        # Merge labels
        feature_df['label'] = feature_df['account_id'].map(labels)
        feature_df = feature_df.dropna(subset=['label'])

        X = feature_df[feature_cols].values
        y = feature_df['label'].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply SMOTE for class imbalance
        smote_config = self.config.get("detection.smote", {})
        smote = SMOTE(
            k_neighbors=smote_config.get("k_neighbors", 5),
            random_state=smote_config.get("random_state", 42)
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # Train XGBoost
        xgb_config = self.config.get("detection.xgboost", {})
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 200),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            random_state=xgb_config.get("random_state", 42),
            eval_metric=xgb_config.get("eval_metric", "mlogloss"),
            early_stopping_rounds=xgb_config.get("early_stopping_rounds", 20),
            verbosity=1
        )

        self.model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)

        # Decode predictions for report
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.label_encoder.classes_)

        # ROC AUC per class
        roc_auc = {}
        for i, cls in enumerate(self.label_encoder.classes_):
            y_binary = (y_test == i).astype(int)
            try:
                roc_auc[cls] = roc_auc_score(y_binary, y_proba[:, i])
            except:
                roc_auc[cls] = 0.0

        self.is_fitted = True
        self.save()

        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist(),
            "roc_auc": roc_auc,
            "n_train": len(X_train_balanced),
            "n_test": len(X_test),
            "feature_importance": self.get_feature_importance()
        }

        logger.info(f"Training complete. Accuracy: {report['accuracy']:.4f}")
        return results

    def predict(self, account_id: str) -> RiskAssessment:
        """Predict mule account probability for a single account."""
        if not self.is_fitted:
            self.load()

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get features for this account
        feature_df, _ = self.prepare_features([account_id])
        if feature_df.empty:
            raise ValueError(f"No features found for account {account_id}")

        X = feature_df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Get probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        # Map to probabilities
        class_proba = dict(zip(self.label_encoder.classes_, proba))
        mule_prob = class_proba.get('MULE', 0) * 100
        suspicious_prob = class_proba.get('SUSPICIOUS', 0) * 100
        legitimate_prob = class_proba.get('LEGITIMATE', 0) * 100

        # Get SHAP explanation
        shap_values = self.shap_explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            # Multi-class: shap_values is a list of arrays, one per class
            shap_values = shap_values[pred_idx]
        # shap_values shape: (1, n_features) - take the first (only) sample
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        # Handle case where shap_values elements might be arrays themselves
        shap_list = []
        for val in shap_values:
            if isinstance(val, (list, np.ndarray)):
                shap_list.append(float(val[0]) if len(val) > 0 else 0.0)
            else:
                shap_list.append(float(val))
        shap_dict = dict(zip(self.feature_names, shap_list))
        top_shap = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        # Determine risk tier from composite score (will be computed by risk engine)
        ml_score = mule_prob * 0.7 + suspicious_prob * 0.3

        # Determine recommended action based on classification
        if pred_label == 'MULE':
            recommended_action = "FREEZE_ACCOUNT"
        elif pred_label == 'SUSPICIOUS':
            recommended_action = "GENERATE_SAR"
        else:
            recommended_action = "MONITOR"

        return RiskAssessment(
            account_id=account_id,
            timestamp=datetime.utcnow(),
            ml_score=ml_score,
            graph_score=0,  # Filled by risk engine
            regulatory_score=0,  # Filled by risk engine
            composite_score=ml_score,  # Will be updated
            risk_tier=RiskTier.WATCH,  # Will be updated
            classification=pred_label,
            mule_probability=mule_prob,
            suspicious_probability=suspicious_prob,
            legitimate_probability=legitimate_prob,
            shap_explanation=top_shap,
            recommended_action=recommended_action
        )

    def predict_batch(self, accounts: List[str]) -> List[RiskAssessment]:
        """Predict for multiple accounts."""
        results = []
        for acc in accounts:
            try:
                results.append(self.predict(acc))
            except Exception as e:
                logger.error(f"Error predicting for {acc}: {e}")
        return results

    def get_shap_explanation(self, account_id: str, top_k: int = 10) -> Dict[str, float]:
        """Get SHAP explanation for a prediction."""
        if not self.is_fitted or self.shap_explainer is None:
            return {}

        feature_df, _ = self.prepare_features([account_id])
        if feature_df.empty:
            return {}

        X = feature_df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)

        shap_values = self.shap_explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            pred_idx = self.model.predict(X_scaled)[0]
            shap_values = shap_values[pred_idx]

        shap_dict = dict(zip(self.feature_names, shap_values[0].tolist()))
        return dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get XGBoost feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def save(self) -> None:
        """Save model to disk."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        joblib.dump(model_data, self.model_dir / "xgboost_classifier.joblib")

        # Save SHAP explainer separately
        if self.shap_explainer:
            joblib.dump(self.shap_explainer, self.model_dir / "shap_explainer.joblib")

        logger.info(f"Classifier saved to {self.model_dir}")

    def load(self) -> bool:
        """Load model from disk."""
        model_path = self.model_dir / "xgboost_classifier.joblib"
        if not model_path.exists():
            return False

        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoder = model_data["label_encoder"]
            self.feature_names = model_data["feature_names"]
            self.is_fitted = model_data["is_fitted"]

            # Load SHAP explainer
            shap_path = self.model_dir / "shap_explainer.joblib"
            if shap_path.exists():
                self.shap_explainer = joblib.load(shap_path)

            logger.info(f"Classifier loaded from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False


def train_full_pipeline():
    """Complete training pipeline for the detection system."""
    from src.ingestion.synthetic_data import generate_all_data
    from src.ingestion.database import get_db

    logger.info("Starting full training pipeline...")

    # Generate and load data
    data = generate_all_data()
    db = get_db()

    # Insert synthetic data
    logger.info("Inserting transactions...")
    db.insert_transactions_batch(data["transactions"])

    logger.info("Inserting FMS alerts...")
    for alert in data["fms_alerts"]:
        db.insert_fms_alert(alert)

    logger.info("Inserting TMS alerts...")
    for alert in data["tms_alerts"]:
        db.insert_tms_alert(alert)

    logger.info("Inserting watchlist entries...")
    for entry in data["watchlist"]:
        db.insert_watchlist_entry(entry)

    # Train anomaly detector
    logger.info("Training anomaly detector...")
    all_accounts = data["accounts"]["legitimate"] + data["accounts"]["mule"] + data["accounts"]["suspicious"] + data["accounts"]["kingpin"]
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(all_accounts)

    # Build graph
    logger.info("Building graph...")
    from src.graph.analysis import GraphAnalyzer
    graph_analyzer = GraphAnalyzer()
    graph_analyzer.build_graph(all_accounts, data["transactions"])

    # Train classifier
    logger.info("Training classifier...")
    classifier = MuleAccountClassifier()
    labels = {}
    for acc in all_accounts:
        if 'MULE' in acc or 'KING' in acc:
            labels[acc] = 'MULE'
        elif 'SUS' in acc:
            labels[acc] = 'SUSPICIOUS'
        else:
            labels[acc] = 'LEGITIMATE'

    results = classifier.fit(all_accounts, labels)

    logger.info("Training pipeline complete!")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_full_pipeline()