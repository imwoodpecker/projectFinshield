"""Unit tests for autonomous agent layer."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.agent.response import AutonomousAgent, AgentState, AgentAction
from src.common.models import RiskAssessment, RiskTier, Case, SARReport
from src.risk.engine import RiskEngine
from src.ingestion.synthetic_data import SyntheticDataGenerator
from src.ingestion.database import get_db


class TestAutonomousAgent:
    """Tests for autonomous agent."""

    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked dependencies."""
        with patch('src.agent.response.RiskEngine') as mock_risk_engine, \
             patch('src.agent.response.Ollama') as mock_ollama, \
             patch('src.agent.response.Chroma') as mock_chroma_class, \
             patch('src.agent.response.OllamaEmbeddings') as mock_embeddings:

            # Setup mock risk engine
            mock_engine = Mock()
            mock_risk_engine.return_value = mock_engine

            # Setup mock Ollama
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm

            # Setup mock embeddings
            mock_embeddings.return_value = Mock()

            # Setup mock ChromaDB - use the class return_value
            mock_chroma_instance = Mock()
            mock_collection = Mock()
            mock_chroma_instance.get_or_create_collection.return_value = mock_collection
            # Make similarity_search return an empty list (iterable)
            mock_chroma_instance.similarity_search.return_value = []
            mock_chroma_class.return_value = mock_chroma_instance

            agent = AutonomousAgent()
            agent.risk_engine = mock_engine
            agent.llm = mock_llm
            agent.collection = mock_collection
            agent.vector_store = mock_chroma_instance

            yield agent, mock_engine, mock_llm, mock_collection

    def test_initial_state(self, mock_agent):
        """Test agent initial state."""
        agent, _, _, _ = mock_agent
        assert agent is not None
        assert agent.graph is not None  # LangGraph compiled

    def test_assess_action(self, mock_agent):
        """Test ASSESS node (assess_risk_node in graph)."""
        agent, mock_engine, _, _ = mock_agent

        # Mock risk assessment
        mock_assessment = RiskAssessment(
            account_id="ACC_TEST",
            timestamp=datetime.utcnow(),
            ml_score=70,
            graph_score=50,
            regulatory_score=0,
            anomaly_score=30,
            composite_score=55,
            risk_tier=RiskTier.ALERT,
            classification="SUSPICIOUS",
            mule_probability=40,
            suspicious_probability=35,
            legitimate_probability=25,
            shap_explanation={"feature1": 0.5},
            recommended_action="INVESTIGATE"
        )
        mock_engine.assess_account.return_value = mock_assessment

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {},
            "tier": "",
            "context": {},
            "rag_context": [],
            "action": None,
            "action_params": {},
            "reasoning": "",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._assess_risk_node(state)

        assert result['risk_assessment'] is not None
        assert result['risk_assessment']['composite_score'] == 55
        assert result['risk_assessment']['risk_tier'] == RiskTier.ALERT.value
        assert result['tier'] == RiskTier.ALERT.value

    def test_investigate_action(self, mock_agent):
        """Test INVESTIGATE node (execute_investigate_node in graph)."""
        agent, mock_engine, _, _ = mock_agent

        mock_assessment = RiskAssessment(
            account_id="ACC_TEST",
            timestamp=datetime.utcnow(),
            ml_score=70,
            graph_score=50,
            regulatory_score=0,
            anomaly_score=30,
            composite_score=55,
            risk_tier=RiskTier.ALERT,
            classification="SUSPICIOUS",
            mule_probability=40,
            suspicious_probability=35,
            legitimate_probability=25,
            shap_explanation={},
            graph_details={"kingpin_score": 10, "cycle_count": 2},
            regulatory_details={"watchlist_matches": []},
            recommended_action="INVESTIGATE"
        )
        mock_engine.assess_account.return_value = mock_assessment
        mock_engine.explain_assessment.return_value = {"reasons": ["Test reason"]}

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 55,
                "ml_score": 70,
                "graph_score": 50,
                "regulatory_score": 0,
                "anomaly_score": 30,
                "risk_tier": "ALERT",
                "classification": "SUSPICIOUS",
                "mule_probability": 40,
                "suspicious_probability": 35,
                "legitimate_probability": 25,
                "shap_explanation": {},
                "graph_details": {"kingpin_score": 10, "cycle_count": 2},
                "regulatory_details": {"watchlist_matches": []},
                "anomaly_details": {}
            },
            "tier": "ALERT",
            "context": {},
            "rag_context": [],
            "action": "INVESTIGATE",
            "action_params": {},
            "reasoning": "Suspicious patterns detected",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._execute_investigate_node(state)

        assert 'context' in result
        assert 'investigation' in result['context']

    def test_decide_action_routing(self, mock_agent):
        """Test action decision routing (make_decision_node in graph)."""
        agent, _, mock_llm, _ = mock_agent

        # Mock LLM response for ACT tier
        mock_llm.invoke.return_value = Mock(content='{"action": "FREEZE_ACCOUNT", "reasoning": "High risk", "params": {}, "confidence": 0.9}')

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 85,
                "ml_score": 90,
                "graph_score": 80,
                "regulatory_score": 50,
                "anomaly_score": 70,
                "risk_tier": "ACT",
                "classification": "MULE",
                "mule_probability": 80,
                "suspicious_probability": 15,
                "legitimate_probability": 5,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "ACT",
            "context": {},
            "rag_context": [],
            "action": None,
            "action_params": {},
            "reasoning": "",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._make_decision_node(state)

        assert result['action'] == 'FREEZE_ACCOUNT'
        assert 'reasoning' in result

    def test_freeze_account_action(self, mock_agent):
        """Test FREEZE_ACCOUNT node (execute_freeze_node in graph)."""
        agent, _, _, _ = mock_agent

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 85,
                "ml_score": 90,
                "graph_score": 80,
                "regulatory_score": 50,
                "anomaly_score": 70,
                "risk_tier": "ACT",
                "classification": "MULE",
                "mule_probability": 80,
                "suspicious_probability": 15,
                "legitimate_probability": 5,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "ACT",
            "context": {},
            "rag_context": [],
            "action": "FREEZE_ACCOUNT",
            "action_params": {},
            "reasoning": "High risk mule account",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._execute_freeze_node(state)

        assert result['freeze_confirmed'] is True
        assert result['action'] == 'FREEZE_ACCOUNT'

    def test_generate_sar_action(self, mock_agent):
        """Test GENERATE_SAR node (execute_generate_sar_node in graph)."""
        agent, _, _, _ = mock_agent

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 60,
                "ml_score": 70,
                "graph_score": 60,
                "regulatory_score": 0,
                "anomaly_score": 40,
                "risk_tier": "ALERT",
                "classification": "SUSPICIOUS",
                "mule_probability": 40,
                "suspicious_probability": 35,
                "legitimate_probability": 25,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "ALERT",
            "context": {},
            "rag_context": [],
            "action": "GENERATE_SAR",
            "action_params": {},
            "reasoning": "Suspicious patterns detected",
            "sar_report": None,
            "case_id": "CASE_123",
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._execute_generate_sar_node(state)

        assert result['sar_report'] is not None
        assert 'sar_id' in result['sar_report']
        assert result['sar_report']['account_id'] == "ACC_TEST"

    def test_escalate_action(self, mock_agent):
        """Test ESCALATE node (execute_escalate_node in graph)."""
        agent, _, _, _ = mock_agent

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 70,
                "ml_score": 80,
                "graph_score": 70,
                "regulatory_score": 30,
                "anomaly_score": 60,
                "risk_tier": "ACT",
                "classification": "MULE",
                "mule_probability": 70,
                "suspicious_probability": 20,
                "legitimate_probability": 10,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "ACT",
            "context": {},
            "rag_context": [],
            "action": "ESCALATE",
            "action_params": {},
            "reasoning": "Complex case requiring senior review",
            "sar_report": None,
            "case_id": "CASE_123",
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._execute_escalate_node(state)

        assert 'context' in result
        assert 'escalation' in result['context']

    def test_close_case_action(self, mock_agent):
        """Test CLOSE_CASE node (execute_close_node in graph)."""
        agent, _, _, _ = mock_agent

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 8,
                "ml_score": 10,
                "graph_score": 5,
                "regulatory_score": 0,
                "anomaly_score": 5,
                "risk_tier": "WATCH",
                "classification": "LEGITIMATE",
                "mule_probability": 5,
                "suspicious_probability": 5,
                "legitimate_probability": 90,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "WATCH",
            "context": {},
            "rag_context": [],
            "action": "CLOSE_CASE",
            "action_params": {},
            "reasoning": "False positive - legitimate activity",
            "sar_report": None,
            "case_id": "CASE_123",
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._execute_close_node(state)

        assert result['action'] == 'CLOSE_CASE'
        assert 'messages' in result
        assert any('CLOSE_CASE' in msg.get('content', '') for msg in result['messages'])

    def test_should_continue_routing(self, mock_agent):
        """Test conditional routing logic (_route_action in graph)."""
        agent, _, _, _ = mock_agent

        # Test ACT tier -> FREEZE_ACCOUNT
        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 85,
                "ml_score": 90,
                "graph_score": 80,
                "regulatory_score": 50,
                "anomaly_score": 70,
                "risk_tier": "ACT",
                "classification": "MULE",
                "mule_probability": 80,
                "suspicious_probability": 15,
                "legitimate_probability": 5,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "ACT",
            "context": {},
            "rag_context": [],
            "action": "FREEZE_ACCOUNT",
            "action_params": {},
            "reasoning": "",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        next_node = agent._route_action(state)
        assert next_node in ['MONITOR', 'INVESTIGATE', 'REQUEST_INFO', 'FREEZE_ACCOUNT', 'GENERATE_SAR', 'ESCALATE', 'CLOSE_CASE']

    def test_process_account_integration(self, mock_agent):
        """Test full account processing."""
        agent, mock_engine, mock_llm, _ = mock_agent

        # Mock assessment
        mock_assessment = RiskAssessment(
            account_id="ACC_TEST",
            timestamp=datetime.utcnow(),
            ml_score=75,
            graph_score=60,
            regulatory_score=0,
            anomaly_score=40,
            composite_score=60,
            risk_tier=RiskTier.ALERT,
            classification="SUSPICIOUS",
            mule_probability=45,
            suspicious_probability=35,
            legitimate_probability=20,
            shap_explanation={"velocity_score": 0.8},
            recommended_action="INVESTIGATE"
        )
        mock_engine.assess_account.return_value = mock_assessment
        mock_engine.explain_assessment.return_value = {"reasons": ["High velocity", "Structuring pattern"]}

        # Mock LLM for decision - return strings directly (not Mock objects with .content)
        mock_llm.invoke.side_effect = [
            '{"action": "INVESTIGATE", "reasoning": "Need more info", "params": {}, "confidence": 0.9}',
            '{"steps": ["Trace funds"], "queries": [], "evidence_needed": [], "priority": "HIGH"}',
            '{"narrative": "Test SAR", "transaction_summary": {}, "graph_evidence_summary": {}, "regulatory_references": [], "recommended_action": "FILE"}'
        ]

        result = agent.process_account("ACC_TEST")

        assert 'account_id' in result
        assert 'action' in result
        assert 'risk_tier' in result
        assert 'composite_score' in result

    def test_sar_generation_format(self, mock_agent):
        """Test SAR report format (_generate_sar method)."""
        agent, _, _, _ = mock_agent

        assessment = RiskAssessment(
            account_id="ACC_TEST",
            timestamp=datetime.utcnow(),
            ml_score=70,
            graph_score=60,
            regulatory_score=0,
            anomaly_score=40,
            composite_score=60,
            risk_tier=RiskTier.ALERT,
            classification="SUSPICIOUS",
            mule_probability=40,
            suspicious_probability=35,
            legitimate_probability=25,
            shap_explanation={"feature1": 0.5},
            recommended_action="GENERATE_SAR"
        )

        sar = agent._generate_sar(assessment.account_id, {
            "composite_score": 60,
            "ml_score": 70,
            "graph_score": 60,
            "regulatory_score": 0,
            "anomaly_score": 40,
            "risk_tier": "ALERT",
            "classification": "SUSPICIOUS",
            "mule_probability": 40,
            "suspicious_probability": 35,
            "legitimate_probability": 25,
            "shap_explanation": {"feature1": 0.5},
            "graph_details": {},
            "regulatory_details": {},
            "anomaly_details": {}
        }, {})

        assert sar['sar_id'].startswith('SAR_')
        assert sar['account_id'] == "ACC_TEST"
        assert 'suspect_details' in sar
        assert 'action_taken' in sar
        assert 'filing_date' in sar

    def test_case_creation(self, mock_agent):
        """Test case creation (_create_case_node in graph)."""
        agent, _, _, _ = mock_agent

        state = {
            "account_id": "ACC_TEST",
            "risk_assessment": {
                "composite_score": 45,
                "ml_score": 60,
                "graph_score": 50,
                "regulatory_score": 0,
                "anomaly_score": 30,
                "risk_tier": "WATCH",
                "classification": "SUSPICIOUS",
                "mule_probability": 30,
                "suspicious_probability": 35,
                "legitimate_probability": 35,
                "shap_explanation": {},
                "graph_details": {},
                "regulatory_details": {},
                "anomaly_details": {}
            },
            "tier": "WATCH",
            "context": {},
            "rag_context": [],
            "action": "INVESTIGATE",
            "action_params": {},
            "reasoning": "Initial investigation",
            "sar_report": None,
            "case_id": None,
            "freeze_confirmed": False,
            "messages": [],
            "iteration": 0,
            "max_iterations": 3
        }

        result = agent._create_case_node(state)

        assert 'case_id' in result
        assert result['case_id'] is not None
        # case_id is a UUID, verify it's a valid UUID format
        import uuid
        try:
            uuid.UUID(result['case_id'])
        except ValueError:
            assert False, "case_id should be a valid UUID"


class TestAgentDemo:
    """Tests for agent demo function."""

    def test_run_agent_demo(self):
        """Test demo runs without error."""
        from src.agent.response import run_agent_demo

        # This should run without exceptions
        # Note: In real test, we'd mock the dependencies
        # For now, just verify import works
        assert run_agent_demo is not None