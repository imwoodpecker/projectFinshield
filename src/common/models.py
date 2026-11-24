"""Shared data models for FinShield."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class TransactionChannel(str, Enum):
    UPI = "UPI"
    NEFT = "NEFT"
    IMPS = "IMPS"
    RTGS = "RTGS"
    ATM = "ATM"
    CARD = "CARD"
    NET_BANKING = "NET_BANKING"


class TransactionType(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


class AlertSource(str, Enum):
    FMS = "FMS"  # Fraud Monitoring System
    TMS = "TMS"  # Transaction Monitoring System
    GOVT = "GOVT"  # Government feed
    INTERNAL = "INTERNAL"


class RiskTier(str, Enum):
    WATCH = "WATCH"
    ALERT = "ALERT"
    ACT = "ACT"
    CRITICAL = "CRITICAL"


class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    CLOSED = "CLOSED"


class FeedbackLabel(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    MISSED_FRAUD = "MISSED_FRAUD"


class Transaction(BaseModel):
    """Financial transaction record."""
    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    amount: float
    currency: str = "INR"
    channel: TransactionChannel
    txn_type: TransactionType
    timestamp: datetime
    beneficiary_account_id: Optional[str] = None
    beneficiary_name: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FMSAlert(BaseModel):
    """Alert from Fraud Monitoring System."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    alert_type: str
    severity: str
    description: str
    source_system: str
    timestamp: datetime
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class TMSAlert(BaseModel):
    """Alert from Transaction Monitoring System."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    rule_id: str
    rule_name: str
    risk_score: float
    timestamp: datetime
    transaction_ids: List[str] = Field(default_factory=list)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class GovernmentWatchlistEntry(BaseModel):
    """Entry from government fraud watchlist."""
    watchlist_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # RBI, FIU-IND, NPCI, MHA
    account_id: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: str  # account, person, entity
    identifiers: Dict[str, str] = Field(default_factory=dict)  # PAN, Aadhaar, etc.
    listed_date: datetime
    reason: str
    severity: str
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class AccountProfile(BaseModel):
    """90-day behavioural baseline profile for an account."""
    account_id: str
    window_start: datetime
    window_end: datetime
    avg_txn_amount: float
    std_txn_amount: float
    txn_count: int
    avg_daily_txn_count: float
    preferred_hours: List[int] = Field(default_factory=list)  # 0-23
    preferred_channels: List[TransactionChannel] = Field(default_factory=list)
    top_beneficiaries: List[str] = Field(default_factory=list)
    dormancy_periods: List[int] = Field(default_factory=list)  # days inactive
    velocity_score: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class AnomalyScore(BaseModel):
    """Anomaly detection results from PyOD models."""
    account_id: str
    timestamp: datetime
    isolation_forest_score: float
    ecod_score: float
    loda_score: float
    ensemble_score: float
    is_anomaly: bool
    features_used: List[str] = Field(default_factory=list)


class GraphMetrics(BaseModel):
    """Graph analysis results for an account."""
    account_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    cycle_count: int
    max_cycle_length: int
    shared_device_count: int
    shared_ip_count: int
    kingpin_score: float
    mule_network_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    """Composite risk assessment for an account."""
    account_id: str
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ml_score: float  # 0-100 from XGBoost
    graph_score: float  # 0-100 from graph analysis
    regulatory_score: float  # 0-100 from watchlist matches
    anomaly_score: float = 0.0  # 0-100 from anomaly detection ensemble
    composite_score: float  # 0-100 weighted
    risk_tier: RiskTier
    classification: str  # MULE, SUSPICIOUS, LEGITIMATE
    mule_probability: float
    suspicious_probability: float
    legitimate_probability: float
    shap_explanation: Optional[Dict[str, float]] = None
    graph_evidence: Optional[GraphMetrics] = None
    watchlist_matches: List[str] = Field(default_factory=list)
    recommended_action: str
    # Additional details for agent decision making
    graph_details: Dict[str, Any] = Field(default_factory=dict)
    regulatory_details: Dict[str, Any] = Field(default_factory=dict)
    anomaly_details: Dict[str, Any] = Field(default_factory=dict)


class SARReport(BaseModel):
    """Suspicious Activity Report."""
    sar_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    risk_assessment_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    filed_at: Optional[datetime] = None
    status: str = "DRAFT"  # DRAFT, REVIEWED, FILED
    narrative: str
    transaction_summary: Dict[str, Any]
    graph_evidence_summary: Dict[str, Any]
    regulatory_references: List[str] = Field(default_factory=list)
    recommended_action: str
    compliance_officer: Optional[str] = None
    fiud_reference: Optional[str] = None

    # Fields used by agent
    report_type: str = "STR"
    filing_date: Optional[datetime] = None
    reporting_entity: str = "FinShield Bank"
    suspect_details: Dict[str, Any] = Field(default_factory=dict)
    suspicious_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    grounds_of_suspicion: List[str] = Field(default_factory=list)
    action_taken: str = "ENHANCED_MONITORING"


class Case(BaseModel):
    """Case management record."""
    case_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    risk_assessment_id: str
    tier: RiskTier
    status: str = "OPEN"  # OPEN, IN_REVIEW, RESOLVED, ESCALATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    sla_deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    feedback: Optional[FeedbackLabel] = None
    feedback_notes: Optional[str] = None
    analyst_id: Optional[str] = None

    # Fields used by agent
    assigned_agent: Optional[str] = None
    updated_at: Optional[datetime] = None
    actions_taken: List[str] = Field(default_factory=list)
    notes: str = ""
    sar_filed: bool = False
    freeze_requested: bool = False


class ModelFeedback(BaseModel):
    """Feedback for model retraining."""
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    case_id: str
    predicted_label: str
    true_label: FeedbackLabel
    features: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """LangGraph agent state."""
    account_id: str
    risk_assessment: Optional[RiskAssessment] = None
    sar_draft: Optional[SARReport] = None
    action_taken: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_step: str = "analyze"
    retry_count: int = 0