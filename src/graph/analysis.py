"""Graph-based mule account detection using NetworkX."""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json

from src.common.models import GraphMetrics, Transaction, TransactionChannel, TransactionType
from src.ingestion.database import get_db
from src.common.config import get_config

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """NetworkX-based graph analysis for mule network detection."""

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.account_nodes: Set[str] = set()
        self.entity_nodes: Dict[str, Set[str]] = defaultdict(set)  # entity_type -> nodes
        self.min_cycle_length = self.config.get("graph.networkx.min_cycle_length", 3)
        self.max_cycle_length = self.config.get("graph.networkx.max_cycle_length", 10)
        self.kingpin_min_in_degree = self.config.get("graph.kingpin.min_in_degree", 5)
        self.shared_entity_types = self.config.get("graph.shared_entity.entity_types", ["device_id", "ip_address", "phone_number"])

    def build_graph(self, accounts: List[str], transactions: List[Transaction]) -> nx.MultiDiGraph:
        """Build account relationship graph from transactions."""
        logger.info(f"Building graph with {len(accounts)} accounts and {len(transactions)} transactions...")

        self.graph.clear()
        self.account_nodes.clear()
        self.entity_nodes.clear()

        # Add account nodes
        for acc in accounts:
            self.graph.add_node(acc, node_type='account')
            self.account_nodes.add(acc)

        # Add transaction edges
        for txn in transactions:
            if txn.beneficiary_account_id and txn.beneficiary_account_id in self.account_nodes:
                self.graph.add_edge(
                    txn.account_id,
                    txn.beneficiary_account_id,
                    amount=txn.amount,
                    channel=txn.channel.value,
                    timestamp=txn.timestamp,
                    txn_type=txn.txn_type.value
                )

            # Add entity nodes (device, IP, phone)
            if txn.device_id:
                entity_id = f"DEV:{txn.device_id}"
                if entity_id not in self.graph:
                    self.graph.add_node(entity_id, node_type='device')
                self.graph.add_edge(txn.account_id, entity_id, relation='used_device')
                self.entity_nodes['device_id'].add(entity_id)

            if txn.ip_address:
                entity_id = f"IP:{txn.ip_address}"
                if entity_id not in self.graph:
                    self.graph.add_node(entity_id, node_type='ip')
                self.graph.add_edge(txn.account_id, entity_id, relation='used_ip')
                self.entity_nodes['ip_address'].add(entity_id)

            if txn.phone_number:
                entity_id = f"PHONE:{txn.phone_number}"
                if entity_id not in self.graph:
                    self.graph.add_node(entity_id, node_type='phone')
                self.graph.add_edge(txn.account_id, entity_id, relation='used_phone')
                self.entity_nodes['phone_number'].add(entity_id)

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def build_graph_from_db(self, days: int = 90) -> nx.MultiDiGraph:
        """Build graph from database transactions."""
        conn = self.db._get_connection()
        df = conn.execute("""
            SELECT * FROM transactions
            WHERE timestamp >= CURRENT_DATE - INTERVAL ? DAYS
        """, [days]).fetchdf()

        accounts = df['account_id'].unique().tolist()
        transactions = []
        for _, row in df.iterrows():
            transactions.append(Transaction(
                transaction_id=row['transaction_id'],
                account_id=row['account_id'],
                amount=row['amount'],
                channel=TransactionChannel(row['channel']),
                txn_type=TransactionType(row['txn_type']),
                timestamp=pd.to_datetime(row['timestamp']),
                beneficiary_account_id=row['beneficiary_account_id'],
                device_id=row['device_id'],
                ip_address=row['ip_address'],
                phone_number=row['phone_number']
            ))

        return self.build_graph(accounts, transactions)

    def find_cycles(self, account_id: Optional[str] = None) -> List[List[str]]:
        """Find circular money flows (cycles) in the graph."""
        cycles = []

        # Convert to simple directed graph for cycle detection
        simple_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if self.graph.nodes[u].get('node_type') == 'account' and \
               self.graph.nodes[v].get('node_type') == 'account':
                simple_graph.add_edge(u, v)

        try:
            if account_id:
                # Find cycles involving specific account
                all_cycles = list(nx.simple_cycles(simple_graph))
                cycles = [c for c in all_cycles if account_id in c and
                          self.min_cycle_length <= len(c) <= self.max_cycle_length]
            else:
                # Find all cycles
                all_cycles = list(nx.simple_cycles(simple_graph))
                cycles = [c for c in all_cycles if
                          self.min_cycle_length <= len(c) <= self.max_cycle_length]
        except nx.NetworkXNoCycle:
            pass
        except Exception as e:
            logger.warning(f"Cycle detection error: {e}")

        return cycles

    def find_all_cycles(self) -> Dict[str, List[List[str]]]:
        """Find cycles for all accounts."""
        all_cycles = self.find_cycles()
        account_cycles = defaultdict(list)

        for cycle in all_cycles:
            for acc in cycle:
                account_cycles[acc].append(cycle)

        return dict(account_cycles)

    def find_shared_entities(self, account_id: str) -> Dict[str, List[str]]:
        """Find accounts sharing devices, IPs, or phones."""
        shared = defaultdict(list)

        for entity_type in self.shared_entity_types:
            for entity in self.entity_nodes.get(entity_type, []):
                if self.graph.has_edge(account_id, entity):
                    # Find other accounts using this entity
                    for other_acc in self.graph.predecessors(entity):
                        if other_acc != account_id and other_acc in self.account_nodes:
                            shared[entity_type].append(other_acc)

        return {k: list(set(v)) for k, v in shared.items()}

    def compute_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute centrality metrics for all account nodes."""
        # Subgraph of only accounts
        account_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if self.graph.nodes[u].get('node_type') == 'account' and \
               self.graph.nodes[v].get('node_type') == 'account':
                account_graph.add_edge(u, v, weight=data.get('amount', 1))

        metrics = {}

        # Degree centrality
        in_degree = nx.in_degree_centrality(account_graph)
        out_degree = nx.out_degree_centrality(account_graph)

        # Betweenness centrality (sample for performance)
        try:
            betweenness = nx.betweenness_centrality(account_graph, k=min(100, len(account_graph)))
        except:
            betweenness = {n: 0 for n in account_graph.nodes()}

        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(account_graph)
        except:
            closeness = {n: 0 for n in account_graph.nodes()}

        # PageRank
        try:
            pagerank = nx.pagerank(account_graph, weight='weight')
        except:
            pagerank = {n: 0 for n in account_graph.nodes()}

        for node in self.account_nodes:
            metrics[node] = {
                'degree_centrality': (in_degree.get(node, 0) + out_degree.get(node, 0)) / 2,
                'in_degree_centrality': in_degree.get(node, 0),
                'out_degree_centrality': out_degree.get(node, 0),
                'betweenness_centrality': betweenness.get(node, 0),
                'closeness_centrality': closeness.get(node, 0),
                'pagerank': pagerank.get(node, 0)
            }

        return metrics

    def detect_kingpins(self) -> List[Dict[str, Any]]:
        """Detect kingpin/aggregator accounts (high in-degree, many mule senders)."""
        kingpins = []

        # Get in-degree for account nodes
        in_degrees = {}
        for node in self.account_nodes:
            if node in self.graph:
                deg = self.graph.in_degree(node)
                in_degrees[node] = deg if isinstance(deg, (int, float)) else (deg[node] if hasattr(deg, '__getitem__') else dict(deg).get(node, 0))
            else:
                in_degrees[node] = 0

        # Filter for high in-degree
        for node, degree in in_degrees.items():
            if degree >= self.kingpin_min_in_degree:
                # Check how many unique senders
                senders = [u for u in self.graph.predecessors(node)
                           if u in self.account_nodes]

                # Calculate kingpin score based on in-degree and sender diversity
                kingpin_score = min(degree / 20.0, 1.0) * 100

                kingpins.append({
                    'account_id': node,
                    'in_degree': degree,
                    'unique_senders': len(senders),
                    'kingpin_score': kingpin_score,
                    'senders': senders
                })

        return sorted(kingpins, key=lambda x: x['kingpin_score'], reverse=True)

    def detect_mule_networks(self) -> List[Dict[str, Any]]:
        """Detect mule networks (clusters of interconnected accounts)."""
        # Find connected components in account subgraph
        account_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if self.graph.nodes[u].get('node_type') == 'account' and \
               self.graph.nodes[v].get('node_type') == 'account':
                account_graph.add_edge(u, v)

        networks = []
        for component in nx.connected_components(account_graph):
            if len(component) >= 3:  # At least 3 accounts
                subgraph = account_graph.subgraph(component)

                # Calculate network density
                density = nx.density(subgraph)

                # Find cliques (fully connected subgraphs)
                cliques = list(nx.find_cliques(subgraph))
                max_clique_size = max(len(c) for c in cliques) if cliques else 0

                # Check for shared entities within network
                shared_entities = 0
                for entity_type in self.shared_entity_types:
                    for entity in self.entity_nodes.get(entity_type, []):
                        connected_accounts = [n for n in self.graph.predecessors(entity)
                                              if n in component]
                        if len(connected_accounts) >= 2:
                            shared_entities += len(connected_accounts) - 1

                mule_score = min(density * 50 + max_clique_size * 10 + shared_entities * 5, 100)

                networks.append({
                    'accounts': list(component),
                    'size': len(component),
                    'density': density,
                    'max_clique_size': max_clique_size,
                    'shared_entities': shared_entities,
                    'mule_network_score': mule_score
                })

        return sorted(networks, key=lambda x: x['mule_network_score'], reverse=True)

    def compute_account_metrics(self, account_id: str) -> GraphMetrics:
        """Compute comprehensive graph metrics for an account."""
        # Cycles
        cycles = self.find_cycles(account_id)
        cycle_count = len(cycles)
        max_cycle_len = max(len(c) for c in cycles) if cycles else 0

        # Shared entities
        shared = self.find_shared_entities(account_id)
        shared_device_count = len(shared.get('device_id', []))
        shared_ip_count = len(shared.get('ip_address', []))

        # Centrality
        centrality = self.compute_centrality_metrics().get(account_id, {})

        # Kingpin score
        kingpin_score = 0
        in_deg = self.graph.in_degree(account_id)
        if isinstance(in_deg, (int, float)):
            in_deg_val = in_deg
        else:
            # NetworkX 3.x returns a view, need to get the value
            # Check if account_id is in the graph
            if account_id in self.graph:
                in_deg_val = in_deg[account_id] if hasattr(in_deg, '__getitem__') else dict(in_deg).get(account_id, 0)
            else:
                in_deg_val = 0
        if in_deg_val >= self.kingpin_min_in_degree:
            kingpin_score = min(in_deg_val / 20.0, 1.0) * 100

        # Mule network participation
        networks = self.detect_mule_networks()
        mule_network_score = 0
        for net in networks:
            if account_id in net['accounts']:
                mule_network_score = net['mule_network_score']
                break

        return GraphMetrics(
            account_id=account_id,
            degree_centrality=centrality.get('degree_centrality', 0),
            betweenness_centrality=centrality.get('betweenness_centrality', 0),
            closeness_centrality=centrality.get('closeness_centrality', 0),
            pagerank=centrality.get('pagerank', 0),
            cycle_count=cycle_count,
            max_cycle_length=max_cycle_len,
            shared_device_count=shared_device_count,
            shared_ip_count=shared_ip_count,
            kingpin_score=kingpin_score,
            mule_network_score=mule_network_score,
            timestamp=datetime.utcnow()
        )

    def get_account_graph_metrics(self, account_id: str) -> Dict[str, float]:
        """Get graph metrics as dictionary for feature enrichment."""
        metrics = self.compute_account_metrics(account_id)
        return {
            'degree_centrality': metrics.degree_centrality,
            'betweenness_centrality': metrics.betweenness_centrality,
            'closeness_centrality': metrics.closeness_centrality,
            'pagerank': metrics.pagerank,
            'cycle_count': metrics.cycle_count,
            'max_cycle_length': metrics.max_cycle_length,
            'shared_device_count': metrics.shared_device_count,
            'shared_ip_count': metrics.shared_ip_count,
            'kingpin_score': metrics.kingpin_score,
            'mule_network_score': metrics.mule_network_score
        }

    def compute_all_metrics(self) -> List[GraphMetrics]:
        """Compute metrics for all accounts."""
        metrics_list = []
        for acc in self.account_nodes:
            metrics_list.append(self.compute_account_metrics(acc))
        return metrics_list

    def save_metrics_to_db(self, metrics_list: List[GraphMetrics]) -> None:
        """Save graph metrics to database."""
        for metrics in metrics_list:
            self.db.insert_graph_metrics(metrics)

    def get_graph_for_visualization(self, account_id: Optional[str] = None, max_nodes: int = 100) -> Dict[str, Any]:
        """Get subgraph for visualization (pyvis)."""
        if account_id:
            # Get ego network
            account_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                if self.graph.nodes[u].get('node_type') == 'account' and \
                   self.graph.nodes[v].get('node_type') == 'account':
                    account_graph.add_edge(u, v)

            # Get neighbors up to 2 hops
            neighbors = set([account_id])
            for _ in range(2):
                new_neighbors = set()
                for n in neighbors:
                    new_neighbors.update(account_graph.predecessors(n))
                    new_neighbors.update(account_graph.successors(n))
                neighbors.update(new_neighbors)

            subgraph = account_graph.subgraph(list(neighbors)[:max_nodes])
        else:
            # Get top connected component
            account_graph = nx.Graph()
            for u, v, data in self.graph.edges(data=True):
                if self.graph.nodes[u].get('node_type') == 'account' and \
                   self.graph.nodes[v].get('node_type') == 'account':
                    account_graph.add_edge(u, v)

            largest_cc = max(nx.connected_components(account_graph), key=len)
            subgraph = account_graph.subgraph(list(largest_cc)[:max_nodes])

        # Format for pyvis
        nodes = []
        for node in subgraph.nodes():
            node_type = self.graph.nodes[node].get('node_type', 'account')
            color = {
                'account': '#00d4aa',
                'device': '#ff6b6b',
                'ip': '#4ecdc4',
                'phone': '#ffe66d'
            }.get(node_type, '#888888')

            nodes.append({
                'id': node,
                'label': node[:15] + '...' if len(node) > 15 else node,
                'color': color,
                'title': f"{node} ({node_type})",
                'size': 20 if node_type == 'account' else 10
            })

        edges = []
        for u, v in subgraph.edges():
            edges.append({
                'from': u,
                'to': v,
                'color': '#666666',
                'width': 1
            })

        return {'nodes': nodes, 'edges': edges}

    def export_graph_gexf(self, path: str) -> None:
        """Export graph to GEXF format for Gephi."""
        nx.write_gexf(self.graph, path)
        logger.info(f"Graph exported to {path}")