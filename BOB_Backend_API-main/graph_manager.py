import networkx as nx

class GraphManager:
    def __init__(self):
        self.graph = nx.Graph()
        print("GraphManager initialized with an empty graph.")

    def add_node(self, node_id: str, node_type: str):
        if self.graph.has_node(node_id):
            return {"status": "error", "message": "Node already exists."}
        self.graph.add_node(node_id, type=node_type, is_fraudulent=False)
        return {"status": "success", "node_id": node_id, "attributes": self.graph.nodes[node_id]}

    def add_edge(self, source_id: str, target_id: str):
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return {"status": "error", "message": "One or both nodes do not exist."}
        self.graph.add_edge(source_id, target_id)
        return {"status": "success", "edge": (source_id, target_id)}

    def flag_node_as_fraud(self, node_id: str):
        if not self.graph.has_node(node_id):
            return {"status": "error", "message": "Node does not exist."}
        self.graph.nodes[node_id]['is_fraudulent'] = True
        return {"status": "success", "node_id": node_id, "attributes": self.graph.nodes[node_id]}

    def check_fraud_risk(self, user_id: str, max_depth: int = 3):
        if not self.graph.has_node(user_id):
            return {"risk": "undetermined", "message": "User does not exist in the graph."}
        fraudulent_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('is_fraudulent')]
        if not fraudulent_nodes:
            return {"risk": "low", "message": "No fraudulent nodes found in the entire graph."}
        for fraud_node in fraudulent_nodes:
            if nx.has_path(self.graph, source=user_id, target=fraud_node):
                path_length = nx.shortest_path_length(self.graph, source=user_id, target=fraud_node)
                if path_length <= max_depth:
                    path = nx.shortest_path(self.graph, source=user_id, target=fraud_node)
                    return {
                        "risk": "high", "message": f"User '{user_id}' is connected to fraudulent node '{fraud_node}'.",
                        "connection_path_length": path_length, "connection_path": " -> ".join(path)
                    }
        return {"risk": "low", "message": f"User '{user_id}' has no close connections to known fraud."}

graph_db = GraphManager()