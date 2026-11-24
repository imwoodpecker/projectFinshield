"""Streamlit Dashboard for FinShield Fraud Detection System."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.models import RiskAssessment, RiskTier
from src.ingestion.database import get_db
from src.common.config import get_config
from src.risk.engine import RiskEngine, WatchlistManager
from src.graph.analysis import GraphAnalyzer
from src.agent.response import AutonomousAgent, run_agent_demo
from src.detection.classifier import train_full_pipeline
from src.ingestion.synthetic_data import generate_all_data
from pathlib import Path

# Page config
st.set_page_config(
    page_title="FinShield - AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #00d4aa; font-weight: bold;}
    .metric-card {background: #1e1e1e; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00d4aa;}
    .risk-act {border-left-color: #ff4444;}
    .risk-alert {border-left-color: #ffaa00;}
    .risk-watch {border-left-color: #00aaff;}
    .risk-normal {border-left-color: #00d4aa;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding: 0 20px;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = get_db()
if 'config' not in st.session_state:
    st.session_state.config = get_config()
if 'risk_engine' not in st.session_state:
    st.session_state.risk_engine = RiskEngine()
if 'graph_analyzer' not in st.session_state:
    st.session_state.graph_analyzer = GraphAnalyzer()
if 'agent' not in st.session_state:
    st.session_state.agent = AutonomousAgent()

logger = logging.getLogger(__name__)


def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header">🛡️ FinShield</div>', unsafe_allow_html=True)
        st.caption("Autonomous Banking Fraud Detection System")
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Controls")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 Account Explorer", "🕸️ Network Graph", "🤖 Agent Actions",
             "📈 Velocity Monitor", "⚖️ Risk Engine", "🏛️ Watchlist", "📋 Cases & SARs",
             "🔬 Model Training", "⚙️ Settings"]
        )

        st.divider()

        # Filters
        st.subheader("Filters")
        risk_tier = st.multiselect(
            "Risk Tier",
            ["ACT", "ALERT", "WATCH", "NORMAL"],
            default=["ACT", "ALERT", "WATCH"]
        )

        days = st.slider("Lookback Days", 1, 180, 30)

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🔄 Rebuild Graph", use_container_width=True):
            with st.spinner("Rebuilding graph..."):
                st.session_state.graph_analyzer.build_graph_from_db(days)
            st.success("Graph rebuilt!")

        if st.button("🤖 Run Agent on High Risk", use_container_width=True):
            with st.spinner("Running autonomous agent..."):
                run_agent_demo()
            st.success("Agent completed!")

        if st.button("🧠 Retrain Models", use_container_width=True):
            with st.spinner("Training models... this may take a few minutes"):
                train_full_pipeline()
            st.success("Models retrained!")

        return page, risk_tier, days


def get_filtered_accounts(risk_tier_filter: List[str]) -> List[str]:
    """Get account IDs filtered by risk tier."""
    all_accounts = st.session_state.db.get_all_account_ids()
    if not risk_tier_filter:
        return all_accounts

    engine = st.session_state.risk_engine
    filtered = []
    for acc in all_accounts:
        try:
            assessment = engine.assess_account(acc)
            if assessment.risk_tier.value in risk_tier_filter:
                filtered.append(acc)
        except:
            pass
    return filtered


def render_overview(risk_tier_filter: List[str], days: int):
    """Render overview dashboard."""
    st.header("📊 Portfolio Overview")

    engine = st.session_state.risk_engine
    summary = engine.get_risk_summary()

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Accounts", summary['total_accounts'])

    with col2:
        act_count = summary['tier_distribution'].get('ACT', 0)
        st.metric("🔴 ACT", act_count, delta=f"{act_count/summary['total_accounts']*100:.1f}%" if summary['total_accounts'] > 0 else "0%")

    with col3:
        alert_count = summary['tier_distribution'].get('ALERT', 0)
        st.metric("🟠 ALERT", alert_count, delta=f"{alert_count/summary['total_accounts']*100:.1f}%" if summary['total_accounts'] > 0 else "0%")

    with col4:
        watch_count = summary['tier_distribution'].get('WATCH', 0)
        st.metric("🔵 WATCH", watch_count, delta=f"{watch_count/summary['total_accounts']*100:.1f}%" if summary['total_accounts'] > 0 else "0%")

    with col5:
        normal_count = summary['tier_distribution'].get('NORMAL', 0)
        st.metric("🟢 NORMAL", normal_count, delta=f"{normal_count/summary['total_accounts']*100:.1f}%" if summary['total_accounts'] > 0 else "0%")

    # Tier distribution chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_df = pd.DataFrame([
            {"Tier": k, "Count": v} for k, v in summary['tier_distribution'].items()
        ])
        if not tier_df.empty:
            fig = px.pie(tier_df, values='Count', names='Tier', hole=0.4,
                         color='Tier',
                         color_discrete_map={'ACT': '#ff4444', 'ALERT': '#ffaa00',
                                             'WATCH': '#00aaff', 'NORMAL': '#00d4aa'})
            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Component Scores")
        scores = summary.get('average_scores', {})
        if scores:
            score_df = pd.DataFrame([
                {"Component": k.replace('_', ' ').title(), "Score": v}
                for k, v in scores.items()
            ])
            fig = px.bar(score_df, x='Component', y='Score', color='Score',
                         color_continuous_scale='RdYlGn_r', range_color=[0, 100])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Top risk accounts
    st.subheader("🚨 Top Risk Accounts")
    top_risk = summary.get('top_risk_accounts', [])
    if top_risk:
        df = pd.DataFrame(top_risk)
        df['composite_score'] = df['composite_score'].round(1)

        # Color code by tier
        def tier_color(tier):
            colors = {'ACT': '🔴', 'ALERT': '🟠', 'WATCH': '🔵', 'NORMAL': '🟢'}
            return colors.get(tier, '')

        df['Tier'] = df['tier'].apply(lambda x: f"{tier_color(x)} {x}")
        st.dataframe(df[['account_id', 'composite_score', 'Tier']],
                     use_container_width=True, hide_index=True)

    # Recent alerts
    st.subheader("📋 Recent Alerts")
    conn = st.session_state.db._get_connection()
    recent_alerts = conn.execute("""
        SELECT * FROM fms_alerts
        ORDER BY timestamp DESC
        LIMIT 10
    """).fetchdf()

    if not recent_alerts.empty:
        recent_alerts['timestamp'] = pd.to_datetime(recent_alerts['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_alerts[['alert_id', 'account_id', 'alert_type', 'severity', 'timestamp']],
                     use_container_width=True, hide_index=True)


def render_account_explorer(risk_tier_filter: List[str], days: int):
    """Render account explorer with detailed view."""
    st.header("🔍 Account Explorer")

    accounts = get_filtered_accounts(risk_tier_filter)

    if not accounts:
        st.warning("No accounts match the selected filters.")
        return

    # Account selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_account = st.selectbox("Select Account", accounts, index=0)
    with col2:
        if st.button("🔍 Deep Assess", use_container_width=True):
            st.session_state['deep_assess'] = selected_account

    if not selected_account:
        return

    # Get assessment
    engine = st.session_state.risk_engine
    assessment = engine.assess_account(selected_account)
    explanation = engine.explain_assessment(selected_account)

    # Account header with risk tier badge
    tier_colors = {
        'ACT': ('🔴 ACT', 'risk-act'),
        'ALERT': ('🟠 ALERT', 'risk-alert'),
        'WATCH': ('🔵 WATCH', 'risk-watch'),
        'NORMAL': ('🟢 NORMAL', 'risk-normal')
    }
    tier_label, tier_class = tier_colors.get(assessment.risk_tier.value, ('⚪ UNKNOWN', ''))

    st.markdown(f"""
    <div class="metric-card {tier_class}">
        <h3>{selected_account}</h3>
        <div style="display: flex; gap: 2rem; margin-top: 1rem;">
            <div><strong>Risk Tier:</strong> {tier_label}</div>
            <div><strong>Composite Score:</strong> {assessment.composite_score:.1f}/100</div>
            <div><strong>ML Classification:</strong> {assessment.classification}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Component scores
    st.subheader("Component Scores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ML Score", f"{assessment.ml_score:.1f}")
    with col2:
        st.metric("Graph Score", f"{assessment.graph_score:.1f}")
    with col3:
        st.metric("Regulatory Score", f"{assessment.regulatory_score:.1f}")
    with col4:
        st.metric("Anomaly Score", f"{assessment.anomaly_score:.1f}")

    # ML Details
    with st.expander("🤖 ML Classification Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            prob_df = pd.DataFrame({
                'Class': ['MULE', 'SUSPICIOUS', 'LEGITIMATE'],
                'Probability': [
                    assessment.mule_probability,
                    assessment.suspicious_probability,
                    assessment.legitimate_probability
                ]
            })
            fig = px.bar(prob_df, x='Class', y='Probability', color='Class',
                         color_discrete_map={'MULE': '#ff4444', 'SUSPICIOUS': '#ffaa00', 'LEGITIMATE': '#00d4aa'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**SHAP Top Features**")
            if assessment.shap_explanation:
                shap_df = pd.DataFrame([
                    {'Feature': k, 'Impact': v}
                    for k, v in assessment.shap_explanation.items()
                ])
                shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                shap_df = shap_df.sort_values('Abs_Impact', ascending=True)
                fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h',
                             color='Impact', color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP explanation not available")

    # Graph Details
    with st.expander("🕸️ Graph Analysis Details"):
        if assessment.graph_details:
            graph_cols = st.columns(4)
            metrics = assessment.graph_details
            with graph_cols[0]:
                st.metric("Kingpin Score", f"{metrics.get('kingpin_score', 0):.1f}")
                st.metric("In-Degree Centrality", f"{metrics.get('in_degree_centrality', 0):.4f}")
            with graph_cols[1]:
                st.metric("Mule Network Score", f"{metrics.get('mule_network_score', 0):.1f}")
                st.metric("Cycle Count", metrics.get('cycle_count', 0))
            with graph_cols[2]:
                st.metric("Shared Devices", metrics.get('shared_device_count', 0))
                st.metric("Shared IPs", metrics.get('shared_ip_count', 0))
            with graph_cols[3]:
                st.metric("PageRank", f"{metrics.get('pagerank', 0):.6f}")
                st.metric("Betweenness", f"{metrics.get('betweenness_centrality', 0):.6f}")

    # Regulatory Details
    with st.expander("🏛️ Regulatory / Watchlist Matches"):
        reg_details = assessment.regulatory_details
        if reg_details.get('watchlist_matches'):
            for match in reg_details['watchlist_matches']:
                st.warning(f"**{match.get('source', 'Unknown')}** - {match.get('entity_name', 'N/A')} "
                          f"({match.get('severity', 'UNKNOWN')}): {match.get('reason', 'No reason provided')}")
        else:
            st.success("No watchlist matches found")

    # Transaction History
    with st.expander("💳 Transaction History"):
        df = st.session_state.db.get_account_transactions(selected_account, days)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Total Volume", f"₹{df['amount'].sum():,.0f}")
            with col3:
                st.metric("Avg Amount", f"₹{df['amount'].mean():,.0f}")

            # Time series
            daily = df.groupby(df['timestamp'].dt.date)['amount'].sum().reset_index()
            fig = px.line(daily, x='timestamp', y='amount', title='Daily Transaction Volume')
            st.plotly_chart(fig, use_container_width=True)

            # Transaction table
            st.dataframe(df[['transaction_id', 'timestamp', 'amount', 'channel', 'txn_type',
                           'beneficiary_account_id', 'device_id']].head(50),
                        use_container_width=True, hide_index=True)
        else:
            st.info("No transactions found")


def render_network_graph(risk_tier_filter: List[str], days: int):
    """Render network graph visualization."""
    st.header("🕸️ Mule Network Visualization")

    # Build graph
    with st.spinner("Building network graph..."):
        graph_data = st.session_state.graph_analyzer.get_graph_for_visualization(max_nodes=200)

    if not graph_data['nodes']:
        st.warning("No graph data available. Generate synthetic data first.")
        return

    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white",
                  directed=True, notebook=False)

    # Add nodes
    for node in graph_data['nodes']:
        net.add_node(node['id'], label=node['label'], color=node['color'],
                     title=node['title'], size=node['size'])

    # Add edges
    for edge in graph_data['edges']:
        net.add_edge(edge['from'], edge['to'], color=edge['color'], width=edge['width'])

    # Physics settings
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 100},
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "edges": {"smooth": {"type": "continuous"}},
        "interaction": {"hover": true, "navigationButtons": true}
    }
    """)

    # Save and display
    html_path = "/tmp/finshield_graph.html"
    net.save_graph(html_path)

    with open(html_path, 'r') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650)

    # Graph statistics
    st.subheader("Graph Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", len(graph_data['nodes']))
    with col2:
        st.metric("Total Edges", len(graph_data['edges']))
    with col3:
        account_nodes = [n for n in graph_data['nodes'] if n['color'] == '#00d4aa']
        st.metric("Account Nodes", len(account_nodes))
    with col4:
        entity_nodes = [n for n in graph_data['nodes'] if n['color'] != '#00d4aa']
        st.metric("Entity Nodes", len(entity_nodes))

    # Detected networks
    st.subheader("Detected Mule Networks")
    networks = st.session_state.graph_analyzer.detect_mule_networks()
    if networks:
        net_df = pd.DataFrame(networks)
        net_df['accounts'] = net_df['accounts'].apply(lambda x: ', '.join(x[:5]) + ('...' if len(x) > 5 else ''))
        st.dataframe(net_df[['accounts', 'size', 'density', 'max_clique_size', 'mule_network_score']],
                     use_container_width=True, hide_index=True)

    # Kingpins
    st.subheader("Detected Kingpins")
    kingpins = st.session_state.graph_analyzer.detect_kingpins()
    if kingpins:
        kp_df = pd.DataFrame(kingpins)
        kp_df['senders'] = kp_df['senders'].apply(lambda x: ', '.join(x[:5]) + ('...' if len(x) > 5 else ''))
        st.dataframe(kp_df[['account_id', 'in_degree', 'unique_senders', 'kingpin_score', 'senders']],
                     use_container_width=True, hide_index=True)


def render_agent_actions(risk_tier_filter: List[str], days: int):
    """Render autonomous agent actions."""
    st.header("🤖 Autonomous Agent Actions")

    agent = st.session_state.agent

    # Get high-risk accounts
    engine = st.session_state.risk_engine
    high_risk = engine.get_high_risk_accounts(RiskTier.ALERT, limit=20)

    if not high_risk:
        st.info("No high-risk accounts to process.")
        return

    # Account selector for agent
    account_options = {f"{a.account_id} ({a.risk_tier.value}, {a.composite_score:.1f})": a.account_id
                       for a in high_risk}

    selected = st.selectbox("Select Account for Agent Processing",
                            options=list(account_options.keys()))
    selected_account = account_options[selected]

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Agent", type="primary", use_container_width=True):
            with st.spinner(f"Processing {selected_account}..."):
                result = agent.process_account(selected_account)
                st.session_state['last_agent_result'] = result
            st.success("Agent completed!")

    with col2:
        if st.button("📊 Run on All High Risk", use_container_width=True):
            with st.spinner("Processing all high-risk accounts..."):
                results = agent.process_batch([a.account_id for a in high_risk])
                st.session_state['batch_agent_results'] = results
            st.success(f"Processed {len(results)} accounts!")

    # Display last result
    if 'last_agent_result' in st.session_state:
        result = st.session_state['last_agent_result']

        st.divider()
        st.subheader(f"Agent Result: {result['account_id']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Action Taken", result['action'])
        with col2:
            st.metric("Risk Tier", result['risk_tier'])
        with col3:
            st.metric("Composite Score", f"{result['composite_score']:.1f}")

        st.write(f"**Reasoning:** {result['reasoning']}")

        if result.get('case_id'):
            st.info(f"Case Created: {result['case_id']}")

        if result.get('sar_report'):
            st.warning(f"SAR Generated: {result['sar_report']['sar_id']}")

        if result.get('freeze_confirmed'):
            st.error("🔴 ACCOUNT FROZEN")

        # Message log
        with st.expander("Agent Message Log"):
            for msg in result.get('messages', []):
                st.text(f"[{msg['role']}] {msg['content']}")

    # Batch results
    if 'batch_agent_results' in st.session_state:
        st.divider()
        st.subheader("Batch Processing Results")

        batch_results = st.session_state['batch_agent_results']
        summary_df = pd.DataFrame([
            {
                'Account': r['account_id'],
                'Action': r['action'],
                'Tier': r['risk_tier'],
                'Score': f"{r['composite_score']:.1f}",
                'Case ID': r.get('case_id', 'N/A'),
                'SAR': r.get('sar_report', {}).get('sar_id', 'No') if r.get('sar_report') else 'No',
                'Frozen': 'Yes' if r.get('freeze_confirmed') else 'No'
            }
            for r in batch_results
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_velocity_monitor(risk_tier_filter: List[str], days: int):
    """Render real-time velocity monitor."""
    st.header("📈 Velocity Monitor")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if auto_refresh:
        st.empty()  # Placeholder for auto-refresh

    # Recent transactions
    col1, col2 = st.columns([1, 3])
    with col1:
        hours = st.selectbox("Time Window", [1, 6, 12, 24], index=0)
    with col2:
        st.caption(f"Showing transactions from last {hours} hour(s)")

    df = st.session_state.db.get_recent_transactions(hours)

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Velocity by account
        st.subheader("Top Accounts by Velocity")
        velocity_df = df.groupby('account_id').agg(
            txn_count=('transaction_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean')
        ).reset_index().sort_values('txn_count', ascending=False).head(20)

        fig = px.bar(velocity_df.head(10), x='account_id', y='txn_count',
                     color='total_amount', color_continuous_scale='Viridis',
                     title='Transaction Count by Account (Last {} Hours)'.format(hours))
        st.plotly_chart(fig, use_container_width=True)

        # Time series of transaction volume
        st.subheader("Transaction Volume Over Time")
        df['time_bucket'] = df['timestamp'].dt.floor('5min')
        time_series = df.groupby('time_bucket').agg(
            volume=('amount', 'sum'),
            count=('transaction_id', 'count')
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=time_series['time_bucket'], y=time_series['count'],
                            name='Txn Count', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_series['time_bucket'], y=time_series['volume'],
                                name='Volume (₹)', line=dict(color='orange')), secondary_y=True)
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Volume (₹)", secondary_y=True)
        fig.update_layout(height=400, title="5-Minute Transaction Velocity")
        st.plotly_chart(fig, use_container_width=True)

        # Channel breakdown
        col1, col2 = st.columns(2)
        with col1:
            channel_counts = df['channel'].value_counts()
            fig = px.pie(values=channel_counts.values, names=channel_counts.index,
                         title='Transactions by Channel')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            txn_type_counts = df['txn_type'].value_counts()
            fig = px.pie(values=txn_type_counts.values, names=txn_type_counts.index,
                         title='Transactions by Type')
            st.plotly_chart(fig, use_container_width=True)

        # Recent transactions table
        st.subheader("Recent Transactions")
        display_df = df.sort_values('timestamp', ascending=False).head(50)
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df[['transaction_id', 'timestamp', 'account_id', 'amount',
                                 'channel', 'txn_type', 'beneficiary_account_id']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No recent transactions")


def render_risk_engine(risk_tier_filter: List[str], days: int):
    """Render risk engine configuration and testing."""
    st.header("⚖️ Risk Engine Configuration")

    engine = st.session_state.risk_engine

    # Current weights
    st.subheader("Scoring Weights")
    weights = engine.weights

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        new_ml = st.slider("ML Weight", 0.0, 1.0, weights.get('ml', 0.4), 0.05)
    with col2:
        new_graph = st.slider("Graph Weight", 0.0, 1.0, weights.get('graph', 0.3), 0.05)
    with col3:
        new_reg = st.slider("Regulatory Weight", 0.0, 1.0, weights.get('regulatory', 0.2), 0.05)
    with col4:
        new_anom = st.slider("Anomaly Weight", 0.0, 1.0, weights.get('anomaly', 0.1), 0.05)

    total = new_ml + new_graph + new_reg + new_anom
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f}, should be 1.0")

    if st.button("Update Weights"):
        engine.weights = {
            'ml': new_ml, 'graph': new_graph,
            'regulatory': new_reg, 'anomaly': new_anom
        }
        st.success("Weights updated!")

    # Thresholds
    st.subheader("Risk Tier Thresholds")
    thresholds = engine.thresholds

    col1, col2, col3 = st.columns(3)
    with col1:
        watch_thresh = st.number_input("WATCH Threshold", 0, 100, thresholds.get('watch', 30))
    with col2:
        alert_thresh = st.number_input("ALERT Threshold", 0, 100, thresholds.get('alert', 60))
    with col3:
        act_thresh = st.number_input("ACT Threshold", 0, 100, thresholds.get('act', 80))

    if st.button("Update Thresholds"):
        engine.thresholds = {'watch': watch_thresh, 'alert': alert_thresh, 'act': act_thresh}
        st.success("Thresholds updated!")

    # Test scoring on sample accounts
    st.subheader("Test Scoring")
    test_accounts = st.session_state.db.get_all_account_ids()[:10]
    test_account = st.selectbox("Test Account", test_accounts)

    if st.button("Run Assessment"):
        with st.spinner("Assessing..."):
            assessment = engine.assess_account(test_account)
            explanation = engine.explain_assessment(test_account)

        st.json({
            "account_id": assessment.account_id,
            "composite_score": assessment.composite_score,
            "risk_tier": assessment.risk_tier.value,
            "ml_score": assessment.ml_score,
            "graph_score": assessment.graph_score,
            "regulatory_score": assessment.regulatory_score,
            "anomaly_score": assessment.anomaly_score,
            "classification": assessment.classification,
            "reasons": explanation['reasons']
        })


def render_watchlist(risk_tier_filter: List[str], days: int):
    """Render watchlist management."""
    st.header("🏛️ Government Watchlist")

    watchlist_mgr = st.session_state.risk_engine.watchlist_manager if hasattr(st.session_state.risk_engine, 'watchlist_manager') else WatchlistManager()

    # Stats
    stats = watchlist_mgr.get_watchlist_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Watchlist Entries", stats['total_entries'])
    with col2:
        sources = [s[0] for s in stats.get('sources', [])]
        st.metric("Sources", len(sources))

    # By source and severity
    if stats['by_source_severity']:
        df = pd.DataFrame(stats['by_source_severity'])
        fig = px.bar(df, x='source', y='count', color='severity',
                     title='Watchlist Entries by Source and Severity',
                     color_discrete_map={'LOW': '#00d4aa', 'MEDIUM': '#00aaff',
                                         'HIGH': '#ffaa00', 'CRITICAL': '#ff4444'})
        st.plotly_chart(fig, use_container_width=True)

    # Upload new watchlist
    st.subheader("Upload Watchlist")
    uploaded_file = st.file_uploader("Choose JSON or CSV file", type=['json', 'csv'])
    source_name = st.text_input("Source Name", "CUSTOM_UPLOAD")

    if uploaded_file and st.button("Process Upload"):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        count = watchlist_mgr.load_from_file(source_name, tmp_path)
        st.success(f"Loaded {count} entries from {source_name}")
        st.rerun()

    # View entries
    st.subheader("Watchlist Entries")
    conn = st.session_state.db._get_connection()
    entries = conn.execute("""
        SELECT watchlist_id, source, account_id, entity_name, entity_type,
               severity, reason, listed_date
        FROM government_watchlist
        ORDER BY listed_date DESC
        LIMIT 100
    """).fetchdf()

    if not entries.empty:
        entries['listed_date'] = pd.to_datetime(entries['listed_date']).dt.strftime('%Y-%m-%d')
        st.dataframe(entries, use_container_width=True, hide_index=True)


def render_cases_sars(risk_tier_filter: List[str], days: int):
    """Render cases and SARs."""
    st.header("📋 Cases & SARs")

    tab1, tab2 = st.tabs(["Cases", "SAR Reports"])

    with tab1:
        conn = st.session_state.db._get_connection()
        cases = conn.execute("""
            SELECT case_id, account_id, risk_tier, status, assigned_agent,
                   created_at, updated_at, sar_filed, freeze_requested
            FROM cases
            ORDER BY created_at DESC
            LIMIT 100
        """).fetchdf()

        if not cases.empty:
            cases['created_at'] = pd.to_datetime(cases['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(cases, use_container_width=True, hide_index=True)

            # Case details
            selected_case = st.selectbox("View Case Details",
                                         options=cases['case_id'].tolist(),
                                         format_func=lambda x: f"Case {x[:8]}...")

            if selected_case:
                case_detail = conn.execute("SELECT * FROM cases WHERE case_id = ?", [selected_case]).fetchone()
                if case_detail:
                    st.json(dict(case_detail))
        else:
            st.info("No cases found")

    with tab2:
        sars = conn.execute("""
            SELECT sar_id, account_id, report_type, filing_date, status,
                   suspect_details, action_taken
            FROM sar_reports
            ORDER BY filing_date DESC
            LIMIT 100
        """).fetchdf()

        if not sars.empty:
            sars['filing_date'] = pd.to_datetime(sars['filing_date']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(sars[['sar_id', 'account_id', 'report_type', 'filing_date', 'status', 'action_taken']],
                         use_container_width=True, hide_index=True)

            # SAR details
            selected_sar = st.selectbox("View SAR Details",
                                        options=sars['sar_id'].tolist())

            if selected_sar:
                sar_detail = conn.execute("SELECT * FROM sar_reports WHERE sar_id = ?", [selected_sar]).fetchone()
                if sar_detail:
                    st.json(dict(sar_detail))
        else:
            st.info("No SAR reports found")


def render_model_training(risk_tier_filter: List[str], days: int):
    """Render model training interface."""
    st.header("🔬 Model Training & Evaluation")

    tab1, tab2, tab3 = st.tabs(["Train Models", "Model Performance", "Feature Importance"])

    with tab1:
        st.subheader("Full Training Pipeline")
        st.write("This will generate synthetic data, train anomaly detector, build graph, and train classifier.")

        if st.button("🚀 Start Full Training", type="primary", use_container_width=True):
            with st.spinner("Running full training pipeline... This may take several minutes."):
                try:
                    results = train_full_pipeline()
                    st.session_state['training_results'] = results
                    st.success("Training completed!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            st.json(results)

    with tab2:
        st.subheader("Model Performance Metrics")

        # Load latest model results
        model_dir = Path("models/classifier")
        if (model_dir / "xgboost_classifier.joblib").exists():
            import joblib
            model_data = joblib.load(model_dir / "xgboost_classifier.joblib")

            if 'classification_report' in model_data:
                report = model_data['classification_report']
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.style.format("{:.3f}"))

            if 'roc_auc' in model_data:
                st.write("**ROC AUC Scores**")
                auc_df = pd.DataFrame(list(model_data['roc_auc'].items()), columns=['Class', 'AUC'])
                st.bar_chart(auc_df.set_index('Class'))

        else:
            st.info("No trained model found. Run training first.")

    with tab3:
        st.subheader("Feature Importance")

        model_dir = Path("models/classifier")
        if (model_dir / "xgboost_classifier.joblib").exists():
            import joblib
            model_data = joblib.load(model_dir / "xgboost_classifier.joblib")

            if 'feature_importance' in model_data and model_data['feature_importance']:
                importance = model_data['feature_importance']
                imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                imp_df = imp_df.sort_values('Importance', ascending=False).head(30)

                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                             title='Top 30 Feature Importances')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available")
        else:
            st.info("No trained model found")


def render_settings(risk_tier_filter: List[str], days: int):
    """Render settings page."""
    st.header("⚙️ System Settings")

    config = st.session_state.config

    # Configuration display
    st.subheader("Current Configuration")
    with st.expander("View Full Config (YAML)"):
        import yaml
        st.code(yaml.dump(config.config), language='yaml')

    # Database info
    st.subheader("Database Status")
    conn = st.session_state.db._get_connection()
    tables = conn.execute("SHOW TABLES").fetchdf()

    if not tables.empty:
        table_stats = []
        for _, row in tables.iterrows():
            table_name = row['name']
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            table_stats.append({"Table": table_name, "Rows": count})

        st.dataframe(pd.DataFrame(table_stats), use_container_width=True, hide_index=True)

    # Generate synthetic data
    st.subheader("Data Generation")
    col1, col2 = st.columns(2)
    with col1:
        n_legitimate = st.number_input("Legitimate Accounts", 100, 10000, 500)
        n_mule = st.number_input("Mule Accounts", 10, 1000, 50)
    with col2:
        n_suspicious = st.number_input("Suspicious Accounts", 10, 1000, 30)
        n_kingpin = st.number_input("Kingpin Accounts", 1, 100, 5)

    if st.button("Generate Synthetic Data", use_container_width=True):
        with st.spinner("Generating synthetic data..."):
            from src.ingestion.synthetic_data import generate_all_data
            data = generate_all_data(
                n_legitimate=n_legitimate,
                n_mule=n_mule,
                n_suspicious=n_suspicious,
                n_kingpin=n_kingpin
            )

            # Insert into database
            st.session_state.db.insert_transactions_batch(data["transactions"])
            for alert in data["fms_alerts"]:
                st.session_state.db.insert_fms_alert(alert)
            for alert in data["tms_alerts"]:
                st.session_state.db.insert_tms_alert(alert)
            for entry in data["watchlist"]:
                st.session_state.db.insert_watchlist_entry(entry)

            st.success(f"Generated {len(data['transactions'])} transactions, "
                      f"{len(data['fms_alerts'])} FMS alerts, "
                      f"{len(data['tms_alerts'])} TMS alerts, "
                      f"{len(data['watchlist'])} watchlist entries")
            st.rerun()

    # System health
    st.subheader("System Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Status", "🟢 Running")
    with col2:
        model_dir = Path("models/classifier/xgboost_classifier.joblib")
        st.metric("ML Model", "🟢 Loaded" if model_dir.exists() else "🔴 Not Trained")
    with col3:
        anomaly_dir = Path("models/anomaly/anomaly_detector.joblib")
        st.metric("Anomaly Model", "🟢 Loaded" if anomaly_dir.exists() else "🔴 Not Trained")


# Main app
def main():
    render_header()
    page, risk_tier_filter, days = render_sidebar()

    # Route to page
    if page == "📊 Overview":
        render_overview(risk_tier_filter, days)
    elif page == "🔍 Account Explorer":
        render_account_explorer(risk_tier_filter, days)
    elif page == "🕸️ Network Graph":
        render_network_graph(risk_tier_filter, days)
    elif page == "🤖 Agent Actions":
        render_agent_actions(risk_tier_filter, days)
    elif page == "📈 Velocity Monitor":
        render_velocity_monitor(risk_tier_filter, days)
    elif page == "⚖️ Risk Engine":
        render_risk_engine(risk_tier_filter, days)
    elif page == "🏛️ Watchlist":
        render_watchlist(risk_tier_filter, days)
    elif page == "📋 Cases & SARs":
        render_cases_sars(risk_tier_filter, days)
    elif page == "🔬 Model Training":
        render_model_training(risk_tier_filter, days)
    elif page == "⚙️ Settings":
        render_settings(risk_tier_filter, days)


if __name__ == "__main__":
    main()