"""FastAPI ingestion API for FinShield."""
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

from src.common.models import (
    Transaction, TransactionChannel, TransactionType,
    FMSAlert, TMSAlert, GovernmentWatchlistEntry
)
from src.ingestion.database import get_db
from src.common.config import get_config

app = FastAPI(
    title="FinShield Ingestion API",
    description="Real-time financial transaction ingestion for fraud detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_db()
config = get_config()


# Request/Response models
class TransactionRequest(BaseModel):
    account_id: str
    amount: float
    currency: str = "INR"
    channel: TransactionChannel
    txn_type: TransactionType
    timestamp: Optional[datetime] = None
    beneficiary_account_id: Optional[str] = None
    beneficiary_name: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class TransactionResponse(BaseModel):
    transaction_id: str
    status: str
    message: str


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]


class FMSAlertRequest(BaseModel):
    account_id: str
    alert_type: str
    severity: str
    description: str
    source_system: str
    timestamp: Optional[datetime] = None
    raw_payload: dict = Field(default_factory=dict)


class TMSAlertRequest(BaseModel):
    account_id: str
    rule_id: str
    rule_name: str
    risk_score: float
    timestamp: Optional[datetime] = None
    transaction_ids: List[str] = Field(default_factory=list)
    raw_payload: dict = Field(default_factory=dict)


class WatchlistRequest(BaseModel):
    source: str
    account_id: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: str
    identifiers: dict = Field(default_factory=dict)
    listed_date: Optional[datetime] = None
    reason: str
    severity: str
    raw_data: dict = Field(default_factory=dict)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "finshield-ingestion", "timestamp": datetime.utcnow().isoformat()}


@app.post("/transactions", response_model=TransactionResponse)
async def ingest_transaction(txn_req: TransactionRequest, background_tasks: BackgroundTasks):
    """Ingest a single transaction."""
    try:
        txn = Transaction(
            transaction_id=str(uuid.uuid4()),
            account_id=txn_req.account_id,
            amount=txn_req.amount,
            currency=txn_req.currency,
            channel=txn_req.channel,
            txn_type=txn_req.txn_type,
            timestamp=txn_req.timestamp or datetime.utcnow(),
            beneficiary_account_id=txn_req.beneficiary_account_id,
            beneficiary_name=txn_req.beneficiary_name,
            device_id=txn_req.device_id,
            ip_address=txn_req.ip_address,
            phone_number=txn_req.phone_number,
            location=txn_req.location,
            metadata=txn_req.metadata
        )

        # Insert in background for high throughput
        background_tasks.add_task(db.insert_transaction, txn)

        return TransactionResponse(
            transaction_id=txn.transaction_id,
            status="accepted",
            message="Transaction queued for processing"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transactions/batch", response_model=dict)
async def ingest_transactions_batch(batch_req: BatchTransactionRequest, background_tasks: BackgroundTasks):
    """Ingest multiple transactions in batch."""
    try:
        transactions = []
        for txn_req in batch_req.transactions:
            txn = Transaction(
                transaction_id=str(uuid.uuid4()),
                account_id=txn_req.account_id,
                amount=txn_req.amount,
                currency=txn_req.currency,
                channel=txn_req.channel,
                txn_type=txn_req.txn_type,
                timestamp=txn_req.timestamp or datetime.utcnow(),
                beneficiary_account_id=txn_req.beneficiary_account_id,
                beneficiary_name=txn_req.beneficiary_name,
                device_id=txn_req.device_id,
                ip_address=txn_req.ip_address,
                phone_number=txn_req.phone_number,
                location=txn_req.location,
                metadata=txn_req.metadata
            )
            transactions.append(txn)

        background_tasks.add_task(db.insert_transactions_batch, transactions)

        return {
            "status": "accepted",
            "count": len(transactions),
            "message": f"{len(transactions)} transactions queued for processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/fms", response_model=dict)
async def ingest_fms_alert(alert_req: FMSAlertRequest, background_tasks: BackgroundTasks):
    """Ingest FMS alert."""
    try:
        alert = FMSAlert(
            alert_id=str(uuid.uuid4()),
            account_id=alert_req.account_id,
            alert_type=alert_req.alert_type,
            severity=alert_req.severity,
            description=alert_req.description,
            source_system=alert_req.source_system,
            timestamp=alert_req.timestamp or datetime.utcnow(),
            raw_payload=alert_req.raw_payload
        )
        background_tasks.add_task(db.insert_fms_alert, alert)
        return {"status": "accepted", "alert_id": alert.alert_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/tms", response_model=dict)
async def ingest_tms_alert(alert_req: TMSAlertRequest, background_tasks: BackgroundTasks):
    """Ingest TMS alert."""
    try:
        alert = TMSAlert(
            alert_id=str(uuid.uuid4()),
            account_id=alert_req.account_id,
            rule_id=alert_req.rule_id,
            rule_name=alert_req.rule_name,
            risk_score=alert_req.risk_score,
            timestamp=alert_req.timestamp or datetime.utcnow(),
            transaction_ids=alert_req.transaction_ids,
            raw_payload=alert_req.raw_payload
        )
        background_tasks.add_task(db.insert_tms_alert, alert)
        return {"status": "accepted", "alert_id": alert.alert_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/watchlist", response_model=dict)
async def ingest_watchlist_entry(wl_req: WatchlistRequest, background_tasks: BackgroundTasks):
    """Ingest government watchlist entry."""
    try:
        entry = GovernmentWatchlistEntry(
            watchlist_id=str(uuid.uuid4()),
            source=wl_req.source,
            account_id=wl_req.account_id,
            entity_name=wl_req.entity_name,
            entity_type=wl_req.entity_type,
            identifiers=wl_req.identifiers,
            listed_date=wl_req.listed_date or datetime.utcnow(),
            reason=wl_req.reason,
            severity=wl_req.severity,
            raw_data=wl_req.raw_data
        )
        background_tasks.add_task(db.insert_watchlist_entry, entry)
        return {"status": "accepted", "watchlist_id": entry.watchlist_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/accounts/{account_id}/transactions")
async def get_account_transactions(account_id: str, days: int = 90):
    """Get transactions for an account."""
    try:
        df = db.get_account_transactions(account_id, days)
        return {"account_id": account_id, "transactions": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/accounts/{account_id}/velocity")
async def get_account_velocity(account_id: str, window_minutes: int = 60):
    """Get transaction velocity for an account."""
    try:
        stats = db.get_velocity_stats(account_id, window_minutes)
        return {"account_id": account_id, "window_minutes": window_minutes, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/accounts/{account_id}/watchlist")
async def check_watchlist(account_id: str):
    """Check if account is on watchlist."""
    try:
        matches = db.is_account_watchlisted(account_id)
        return {"account_id": account_id, "watchlisted": len(matches) > 0, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transactions/recent")
async def get_recent_transactions(hours: int = 1):
    """Get recent transactions across all accounts."""
    try:
        df = db.get_recent_transactions(hours)
        return {"transactions": df.to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For direct running
def run():
    import uvicorn
    uvicorn.run(
        "src.ingestion.api:app",
        host=config.get("ingestion.api.host", "0.0.0.0"),
        port=config.get("ingestion.api.port", 8000),
        workers=config.get("ingestion.api.workers", 4),
        reload=config.get("app.debug", True)
    )


if __name__ == "__main__":
    run()