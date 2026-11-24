"""Autonomous AI Response Layer - LangGraph state machine with Ollama/Llama3, RAG, and SAR generation."""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import uuid
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.models import RiskAssessment, RiskTier, SARReport, Case, AgentState
from src.ingestion.database import get_db
from src.common.config import get_config
from src.risk.engine import RiskEngine, WatchlistManager

logger = logging.getLogger(__name__)


class AgentAction(str, Enum):
    """Possible agent actions."""
    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    REQUEST_INFO = "REQUEST_INFO"
    FREEZE_ACCOUNT = "FREEZE_ACCOUNT"
    GENERATE_SAR = "GENERATE_SAR"
    ESCALATE = "ESCALATE"
    CLOSE_CASE = "CLOSE_CASE"


class ResponseState(TypedDict):
    """State for the LangGraph agent."""
    account_id: str
    risk_assessment: Dict[str, Any]
    tier: str
    context: Dict[str, Any]
    rag_context: List[str]
    action: Optional[str]
    action_params: Dict[str, Any]
    reasoning: str
    sar_report: Optional[Dict[str, Any]]
    case_id: Optional[str]
    freeze_confirmed: bool
    messages: List[Dict[str, str]]
    iteration: int
    max_iterations: int


class AutonomousAgent:
    """LangGraph-based autonomous agent for fraud response."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.risk_engine = RiskEngine()
        self.watchlist_manager = WatchlistManager()

        # Ollama LLM
        self.llm = Ollama(
            model=self.config.get("agent.ollama.model", "llama3.1:8b"),
            base_url=self.config.get("agent.ollama.base_url", "http://localhost:11434"),
            temperature=self.config.get("agent.ollama.temperature", 0.1),
            num_predict=self.config.get("agent.ollama.num_predict", 512)
        )

        # Embeddings for RAG
        self.embeddings = OllamaEmbeddings(
            model=self.config.get("agent.ollama.embedding_model", "nomic-embed-text"),
            base_url=self.config.get("agent.ollama.base_url", "http://localhost:11434")
        )

        # Vector store for RAG
        self.vector_store = self._init_vector_store()

        # Memory for checkpointing
        self.memory = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

        # Prompt templates
        self.decision_prompt = self._create_decision_prompt()
        self.sar_prompt = self._create_sar_prompt()
        self.investigation_prompt = self._create_investigation_prompt()

    def _init_vector_store(self) -> Chroma:
        """Initialize ChromaDB vector store for RAG."""
        persist_dir = self.config.get("agent.chroma.persist_directory", "./data/chromadb")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name="finshield_knowledge"
        )

        # Load knowledge base if empty
        if vector_store._collection.count() == 0:
            self._load_knowledge_base(vector_store)

        return vector_store

    def _load_knowledge_base(self, vector_store: Chroma) -> None:
        """Load regulatory and investigative knowledge into vector store."""
        knowledge_docs = [
            Document(
                page_content="""SAR Filing Requirements (India - FIU-IND):
- Threshold: Cash transactions ≥ ₹10 lakh
- Suspicious transactions: Any transaction that appears unusual, has no clear economic purpose, or indicates money laundering
- Timeline: File within 7 days of detection
- Required fields: Account details, transaction details, suspicion grounds, action taken
- STR (Suspicious Transaction Report) vs SAR (Suspicious Activity Report)""",
                metadata={"category": "regulatory", "source": "FIU-IND guidelines"}
            ),
            Document(
                page_content="""Mule Account Indicators:
- Rapid movement of funds through multiple accounts
- Account opened recently with high velocity
- Shared devices/IPs across multiple accounts
- Structuring: amounts just below reporting thresholds
- Dormant accounts suddenly activated
- Round number transactions
- Circular flows (money returns to originator)
- Kingpin accounts receiving from many mules""",
                metadata={"category": "investigation", "source": "FinShield patterns"}
            ),
            Document(
                page_content="""Account Freeze Procedures:
1. Internal authorization required (Risk tier ACT or above)
2. Document rationale with evidence
3. Notify compliance officer
4. Freeze via core banking API (simulated)
5. Log freeze action with timestamp and operator
6. Preserve transaction evidence
7. Initiate SAR filing within 24 hours""",
                metadata={"category": "procedure", "source": "FinShield SOPs"}
            ),
            Document(
                page_content="""RBI Master Directions on KYC/AML:
- Customer due diligence at onboarding
- Ongoing monitoring of transactions
- Enhanced due diligence for high-risk customers
- Record keeping for 5 years post-relationship
- Reporting to FIU-IND for suspicious transactions
- Internal audit of AML systems annually""",
                metadata={"category": "regulatory", "source": "RBI guidelines"}
            ),
            Document(
                page_content="""Investigation Best Practices:
- Follow the money: trace fund flows across accounts
- Identify ultimate beneficiaries
- Check for shared infrastructure (device, IP, phone)
- Look for layering: multiple hops to obscure origin
- Document timeline of suspicious activity
- Correlate with external intelligence (watchlists, news)
- Preserve chain of custody for evidence""",
                metadata={"category": "investigation", "source": "AML best practices"}
            )
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(knowledge_docs)
        vector_store.add_documents(splits)
        vector_store.persist()
        logger.info(f"Loaded {len(splits)} knowledge chunks into vector store")

    def _create_decision_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["account_id", "composite_score", "classification", "graph_details", "regulatory_details", "anomaly_details", "tier", "context", "rag_context"],
            template="""You are an autonomous fraud response agent for a bank. Analyze the risk assessment and determine the appropriate action.

Account: {account_id}
Risk Tier: {tier}
Composite Score: {composite_score:.1f}
ML Classification: {classification}
Graph Details: {graph_details}
Regulatory Details: {regulatory_details}
Anomaly Details: {anomaly_details}

Relevant Knowledge:
{rag_context}

Available Actions:
- MONITOR: Continue watching, no immediate action
- INVESTIGATE: Deep dive, gather more evidence
- REQUEST_INFO: Ask for additional data from internal systems
- FREEZE_ACCOUNT: Block account (requires ACT tier or strong evidence)
- GENERATE_SAR: Create Suspicious Activity Report
- ESCALATE: Send to senior analyst/compliance
- CLOSE_CASE: No fraud detected, close

Respond in JSON format:
{{
    "action": "ACTION_NAME",
    "reasoning": "Detailed explanation of why this action",
    "params": {{"key": "value"}},
    "confidence": 0.0-1.0
}}"""
        )

    def _create_sar_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["account_id", "risk_assessment", "investigation_findings"],
            template="""Generate a Suspicious Activity Report (SAR) for FIU-IND filing.

Account: {account_id}
Risk Assessment: {risk_assessment}
Investigation Findings: {investigation_findings}

Create a structured SAR with:
1. Reporting entity details
2. Suspect account/entity details
3. Transaction summary (suspicious transactions)
4. Grounds of suspicion
5. Action taken by bank
6. Supporting documents

Respond in JSON format matching SAR schema."""
        )

    def _create_investigation_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["account_id", "risk_assessment", "current_findings"],
            template="""You are investigating a potential mule account. Current findings:
Account: {account_id}
Risk Assessment: {risk_assessment}
Current Findings: {current_findings}

What additional investigation steps should be taken? What evidence to gather?
Suggest specific queries, data sources, or analysis.

Respond in JSON format:
{{
    "steps": ["step1", "step2", ...],
    "queries": ["sql or api queries"],
    "evidence_needed": ["type1", "type2"],
    "priority": "HIGH/MEDIUM/LOW"
}}"""
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(ResponseState)

        # Nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        workflow.add_node("make_decision", self._make_decision_node)
        workflow.add_node("execute_monitor", self._execute_monitor_node)
        workflow.add_node("execute_investigate", self._execute_investigate_node)
        workflow.add_node("execute_request_info", self._execute_request_info_node)
        workflow.add_node("execute_freeze", self._execute_freeze_node)
        workflow.add_node("execute_generate_sar", self._execute_generate_sar_node)
        workflow.add_node("execute_escalate", self._execute_escalate_node)
        workflow.add_node("execute_close", self._execute_close_node)
        workflow.add_node("create_case", self._create_case_node)
        workflow.add_node("log_action", self._log_action_node)

        # Entry point
        workflow.set_entry_point("retrieve_context")

        # Edges
        workflow.add_edge("retrieve_context", "assess_risk")
        workflow.add_edge("assess_risk", "make_decision")

        # Decision routing
        workflow.add_conditional_edges(
            "make_decision",
            self._route_action,
            {
                "MONITOR": "execute_monitor",
                "INVESTIGATE": "execute_investigate",
                "REQUEST_INFO": "execute_request_info",
                "FREEZE_ACCOUNT": "execute_freeze",
                "GENERATE_SAR": "execute_generate_sar",
                "ESCALATE": "execute_escalate",
                "CLOSE_CASE": "execute_close"
            }
        )

        # All action nodes go to create_case then log_action then END
        for action_node in ["execute_monitor", "execute_investigate", "execute_request_info",
                            "execute_freeze", "execute_generate_sar", "execute_escalate", "execute_close"]:
            workflow.add_edge(action_node, "create_case")
        workflow.add_edge("create_case", "log_action")
        workflow.add_edge("log_action", END)

        return workflow.compile(checkpointer=self.memory)

    def _route_action(self, state: ResponseState) -> str:
        """Route to appropriate action node - returns action name for conditional edges mapping."""
        action = state.get("action", "MONITOR")
        # Return the action name (key in conditional edges mapping)
        valid_actions = ["MONITOR", "INVESTIGATE", "REQUEST_INFO", "FREEZE_ACCOUNT", "GENERATE_SAR", "ESCALATE", "CLOSE_CASE"]
        return action if action in valid_actions else "MONITOR"

    def _retrieve_context_node(self, state: ResponseState) -> ResponseState:
        """Retrieve RAG context from vector store."""
        account_id = state["account_id"]
        tier = state["tier"]

        # Query vector store for relevant knowledge
        query = f"fraud response {tier} tier mule account investigation SAR freeze"
        docs = self.vector_store.similarity_search(query, k=5)
        rag_context = [doc.page_content for doc in docs]

        return {**state, "rag_context": rag_context}

    def _assess_risk_node(self, state: ResponseState) -> ResponseState:
        """Get fresh risk assessment."""
        account_id = state["account_id"]
        assessment = self.risk_engine.assess_account(account_id)

        risk_dict = {
            "composite_score": assessment.composite_score,
            "ml_score": assessment.ml_score,
            "graph_score": assessment.graph_score,
            "regulatory_score": assessment.regulatory_score,
            "anomaly_score": assessment.anomaly_score,
            "risk_tier": assessment.risk_tier.value,
            "classification": assessment.classification,
            "mule_probability": assessment.mule_probability,
            "suspicious_probability": assessment.suspicious_probability,
            "legitimate_probability": assessment.legitimate_probability,
            "shap_explanation": assessment.shap_explanation,
            "graph_details": assessment.graph_details,
            "regulatory_details": assessment.regulatory_details,
            "anomaly_details": assessment.anomaly_details
        }

        return {**state, "risk_assessment": risk_dict, "tier": assessment.risk_tier.value}

    def _make_decision_node(self, state: ResponseState) -> ResponseState:
        """Use LLM to decide on action."""
        ra = state["risk_assessment"]
        prompt = self.decision_prompt.format(
            account_id=state["account_id"],
            composite_score=ra.get("composite_score", 0),
            classification=ra.get("classification", "UNKNOWN"),
            graph_details=json.dumps(ra.get("graph_details", {}), default=str),
            regulatory_details=json.dumps(ra.get("regulatory_details", {}), default=str),
            anomaly_details=json.dumps(ra.get("anomaly_details", {}), default=str),
            tier=state["tier"],
            context=json.dumps(state.get("context", {}), default=str),
            rag_context="\n\n".join(state["rag_context"])
        )

        try:
            response = self.llm.invoke(prompt)
            decision = json.loads(response)

            return {
                **state,
                "action": decision.get("action", "MONITOR"),
                "action_params": decision.get("params", {}),
                "reasoning": decision.get("reasoning", ""),
                "iteration": state.get("iteration", 0) + 1
            }
        except Exception as e:
            logger.error(f"Decision error: {e}")
            # Fallback logic based on tier
            tier = state["tier"]
            if tier == "ACT":
                action = "FREEZE_ACCOUNT"
            elif tier == "ALERT":
                action = "INVESTIGATE"
            elif tier == "WATCH":
                action = "MONITOR"
            else:
                action = "CLOSE_CASE"

            return {
                **state,
                "action": action,
                "action_params": {},
                "reasoning": f"Fallback decision based on tier {tier}: {action}",
                "iteration": state.get("iteration", 0) + 1
            }

    def _execute_monitor_node(self, state: ResponseState) -> ResponseState:
        """Execute monitoring action."""
        logger.info(f"MONITOR: Continuing surveillance on {state['account_id']}")
        return {**state, "messages": state.get("messages", []) + [{
            "role": "agent", "content": f"MONITOR: Account {state['account_id']} placed under enhanced monitoring. Reason: {state['reasoning']}"
        }]}

    def _execute_investigate_node(self, state: ResponseState) -> ResponseState:
        """Execute investigation action."""
        account_id = state["account_id"]

        # Get investigation steps from LLM
        prompt = self.investigation_prompt.format(
            account_id=account_id,
            risk_assessment=json.dumps(state["risk_assessment"], default=str),
            current_findings=json.dumps(state.get("context", {}), default=str)
        )

        try:
            response = self.llm.invoke(prompt)
            investigation = json.loads(response)
            steps = investigation.get("steps", [])
        except:
            steps = [
                "Trace all fund flows for last 90 days",
                "Identify all counterparties",
                "Check for shared devices/IPs/phones",
                "Review onboarding documents",
                "Cross-reference with watchlists"
            ]

        # Perform automated investigation
        findings = self._run_investigation(account_id)

        return {
            **state,
            "context": {**state.get("context", {}), "investigation": findings, "llm_steps": steps},
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"INVESTIGATE: Completed investigation for {account_id}. Findings: {json.dumps(findings, default=str)}"
            }]
        }

    def _execute_request_info_node(self, state: ResponseState) -> ResponseState:
        """Request additional information."""
        params = state.get("action_params", {})
        info_type = params.get("info_type", "transaction_history")

        logger.info(f"REQUEST_INFO: Requesting {info_type} for {state['account_id']}")

        # Simulate requesting info from internal systems
        additional_info = self._fetch_additional_info(state["account_id"], info_type)

        return {
            **state,
            "context": {**state.get("context", {}), "additional_info": additional_info},
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"REQUEST_INFO: Retrieved {info_type} for {state['account_id']}"
            }]
        }

    def _execute_freeze_node(self, state: ResponseState) -> ResponseState:
        """Execute account freeze (simulated)."""
        account_id = state["account_id"]

        # Simulate freeze via core banking API
        freeze_result = self._simulate_freeze(account_id, state["reasoning"])

        # Generate SAR after freeze
        sar = self._generate_sar(account_id, state["risk_assessment"], {"freeze_result": freeze_result})

        return {
            **state,
            "action_params": {**state.get("action_params", {}), "freeze_result": freeze_result},
            "freeze_confirmed": freeze_result.get("success", False),
            "sar_report": sar,
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"FREEZE_ACCOUNT: Account {account_id} frozen. SAR generated: {sar.get('sar_id')}"
            }]
        }

    def _execute_generate_sar_node(self, state: ResponseState) -> ResponseState:
        """Generate SAR without freezing."""
        account_id = state["account_id"]
        sar = self._generate_sar(account_id, state["risk_assessment"], {})

        return {
            **state,
            "sar_report": sar,
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"GENERATE_SAR: SAR created for {account_id}: {sar.get('sar_id')}"
            }]
        }

    def _execute_escalate_node(self, state: ResponseState) -> ResponseState:
        """Escalate to senior analyst."""
        logger.info(f"ESCALATE: Escalating {state['account_id']} to senior analyst")

        escalation = {
            "account_id": state["account_id"],
            "risk_assessment": state["risk_assessment"],
            "reasoning": state["reasoning"],
            "escalated_at": datetime.utcnow().isoformat(),
            "escalated_to": "senior_fraud_analyst"
        }

        # Save escalation
        self.db.save_escalation(escalation)

        return {
            **state,
            "context": {**state.get("context", {}), "escalation": escalation},
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"ESCALATE: {state['account_id']} escalated to senior analyst. Reason: {state['reasoning']}"
            }]
        }

    def _execute_close_node(self, state: ResponseState) -> ResponseState:
        """Close case - no fraud detected."""
        logger.info(f"CLOSE_CASE: Closing case for {state['account_id']}")

        return {
            **state,
            "messages": state.get("messages", []) + [{
                "role": "agent", "content": f"CLOSE_CASE: No fraud detected for {state['account_id']}. Reason: {state['reasoning']}"
            }]
        }

    def _create_case_node(self, state: ResponseState) -> ResponseState:
        """Create or update case record."""
        account_id = state["account_id"]
        tier = state["tier"]
        action = state["action"]

        case = Case(
            case_id=str(uuid.uuid4()),
            account_id=account_id,
            risk_assessment_id=str(uuid.uuid4()),
            tier=RiskTier(tier),
            status="OPEN" if action != "CLOSE_CASE" else "CLOSED"
        )

        self.db.insert_case(case)

        return {**state, "case_id": case.case_id}

    def _log_action_node(self, state: ResponseState) -> ResponseState:
        """Log agent action to database."""
        log_entry = {
            "account_id": state["account_id"],
            "case_id": state.get("case_id"),
            "action": state["action"],
            "reasoning": state["reasoning"],
            "risk_assessment": state["risk_assessment"],
            "timestamp": datetime.utcnow().isoformat(),
            "agent_version": "1.0"
        }
        self.db.log_agent_action(log_entry)
        return state

    def _run_investigation(self, account_id: str) -> Dict[str, Any]:
        """Run automated investigation steps."""
        findings = {}

        # Get transaction flow
        df = self.db.get_account_transactions(account_id, days=90)
        if not df.empty:
            findings["transaction_count"] = len(df)
            findings["total_volume"] = float(df['amount'].sum())
            findings["unique_counterparties"] = df['beneficiary_account_id'].nunique()
            findings["channels_used"] = df['channel'].unique().tolist()
            findings["date_range"] = {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }

        # Check shared entities
        related = self.watchlist_manager.cross_channel_identity_resolution(account_id)
        findings["related_accounts"] = related
        findings["related_count"] = len(related)

        # Get velocity
        velocity = self.db.get_velocity_stats(account_id, window_minutes=60)
        findings["velocity"] = velocity

        # Check watchlist
        watchlist = self.db.is_account_watchlisted(account_id)
        findings["watchlist_matches"] = watchlist

        return findings

    def _fetch_additional_info(self, account_id: str, info_type: str) -> Dict[str, Any]:
        """Fetch additional info from internal systems (simulated)."""
        if info_type == "transaction_history":
            df = self.db.get_account_transactions(account_id, days=180)
            return {"transactions": df.to_dict(orient="records") if not df.empty else []}
        elif info_type == "kyc_documents":
            return {"kyc_status": "VERIFIED", "documents": ["AADHAAR", "PAN", "ADDRESS_PROOF"]}
        elif info_type == "device_fingerprint":
            df = self.db.get_account_transactions(account_id, days=30)
            devices = df['device_id'].dropna().unique().tolist() if not df.empty else []
            return {"devices": devices}
        return {}

    def _simulate_freeze(self, account_id: str, reason: str) -> Dict[str, Any]:
        """Simulate account freeze via core banking API."""
        freeze_id = str(uuid.uuid4())
        result = {
            "freeze_id": freeze_id,
            "account_id": account_id,
            "status": "FROZEN",
            "reason": reason,
            "frozen_at": datetime.utcnow().isoformat(),
            "frozen_by": "autonomous_agent",
            "api_response": {"code": 200, "message": "Account frozen successfully"},
            "success": True
        }

        # Save freeze record
        self.db.save_freeze(result)

        return result

    def _generate_sar(self, account_id: str, risk_assessment: Dict, context: Dict) -> Dict[str, Any]:
        """Generate SAR report."""
        sar_id = f"SAR_{datetime.utcnow().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"

        # Get suspicious transactions
        df = self.db.get_account_transactions(account_id, days=90)
        suspicious_txns = []
        if not df.empty:
            # Flag high-risk transactions
            for _, row in df.iterrows():
                if row['amount'] > 100000:  # >1 lakh
                    suspicious_txns.append({
                        "transaction_id": row['transaction_id'],
                        "amount": float(row['amount']),
                        "timestamp": row['timestamp'],
                        "channel": row['channel'],
                        "counterparty": row['beneficiary_account_id'],
                        "reason": "High value transaction"
                    })

        sar = SARReport(
            sar_id=sar_id,
            account_id=account_id,
            risk_assessment_id=risk_assessment.get("assessment_id", str(uuid.uuid4())),
            report_type="STR",
            filing_date=datetime.utcnow(),
            reporting_entity="FinShield Bank",
            suspect_details={
                "account_id": account_id,
                "risk_tier": risk_assessment.get("risk_tier"),
                "composite_score": risk_assessment.get("composite_score"),
                "classification": risk_assessment.get("classification")
            },
            suspicious_transactions=suspicious_txns,
            grounds_of_suspicion=[
                f"ML Classification: {risk_assessment.get('classification')}",
                f"Composite Risk Score: {risk_assessment.get('composite_score', 0):.1f}",
                f"Graph Risk: {risk_assessment.get('graph_score', 0):.1f}",
                f"Regulatory Matches: {len(risk_assessment.get('regulatory_details', {}).get('watchlist_matches', []))}"
            ],
            action_taken="ACCOUNT_FROZEN" if context.get("freeze_result", {}).get("success") else "ENHANCED_MONITORING",
            status="DRAFT",
            narrative=f"Suspicious activity detected for account {account_id}. Classification: {risk_assessment.get('classification')}, Composite Score: {risk_assessment.get('composite_score', 0):.1f}",
            transaction_summary={"suspicious_count": len(suspicious_txns), "total_amount": sum(t['amount'] for t in suspicious_txns)},
            graph_evidence_summary=risk_assessment.get("graph_details", {}),
            regulatory_references=[],
            recommended_action="FILE",
            compliance_officer="autonomous_agent"
        )

        self.db.insert_sar_report(sar)

        return {
            "sar_id": sar_id,
            "account_id": account_id,
            "status": "DRAFT",
            "generated_at": datetime.utcnow().isoformat(),
            "filing_date": datetime.utcnow().isoformat(),
            "suspect_details": sar.suspect_details,
            "action_taken": sar.action_taken
        }

    def process_account(self, account_id: str, context: Dict = None) -> Dict[str, Any]:
        """Process a single account through the agent workflow."""
        initial_state = ResponseState(
            account_id=account_id,
            risk_assessment={},
            tier="",
            context=context or {},
            rag_context=[],
            action=None,
            action_params={},
            reasoning="",
            sar_report=None,
            case_id=None,
            freeze_confirmed=False,
            messages=[],
            iteration=0,
            max_iterations=3
        )

        config = {"configurable": {"thread_id": f"agent_{account_id}_{datetime.utcnow().timestamp()}"}}
        result = self.graph.invoke(initial_state, config)

        return {
            "account_id": account_id,
            "action": result["action"],
            "reasoning": result["reasoning"],
            "risk_tier": result["tier"],
            "composite_score": result["risk_assessment"].get("composite_score", 0),
            "case_id": result.get("case_id"),
            "sar_report": result.get("sar_report"),
            "freeze_confirmed": result.get("freeze_confirmed", False),
            "messages": result.get("messages", [])
        }

    def process_batch(self, account_ids: List[str]) -> List[Dict[str, Any]]:
        """Process multiple accounts."""
        results = []
        for acc_id in account_ids:
            try:
                results.append(self.process_account(acc_id))
            except Exception as e:
                logger.error(f"Error processing {acc_id}: {e}")
                results.append({"account_id": acc_id, "error": str(e)})
        return results


def run_agent_demo():
    """Demo the autonomous agent."""
    logging.basicConfig(level=logging.INFO)

    agent = AutonomousAgent()
    engine = RiskEngine()

    # Get high-risk accounts
    high_risk = engine.get_high_risk_accounts(RiskTier.ALERT, limit=3)

    print(f"Processing {len(high_risk)} high-risk accounts...\n")

    for assessment in high_risk:
        print(f"\n{'='*60}")
        print(f"Account: {assessment.account_id}")
        print(f"Tier: {assessment.risk_tier.value}, Score: {assessment.composite_score:.1f}")

        result = agent.process_account(assessment.account_id)

        print(f"Action: {result['action']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Case ID: {result.get('case_id')}")
        if result.get('sar_report'):
            print(f"SAR: {result['sar_report']['sar_id']}")
        if result.get('freeze_confirmed'):
            print("ACCOUNT FROZEN")


if __name__ == "__main__":
    run_agent_demo()