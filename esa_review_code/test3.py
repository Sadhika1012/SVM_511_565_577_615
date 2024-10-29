import community
from neo4j import GraphDatabase
import networkx as nx
from networkx.algorithms import community

# Neo4j connection details
uri = "neo4j+ssc://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

# Define connection function
def connect_to_neo4j():
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to fetch graph data from Neo4j and create a NetworkX graph
def fetch_graph_from_neo4j():
    driver = connect_to_neo4j()
    G = nx.Graph()
    
    with driver.session() as session:
        # Query to retrieve usernames and relationships
        query = """
        MATCH (n:Net)-[r]->(m:Net)
        RETURN n.username AS source, m.username AS target, type(r) AS relationship
        """
        
        results = session.run(query)
        for record in results:
            source = record["source"]
            target = record["target"]
            G.add_edge(source, target)
    
    driver.close()
    return G

def calculate_degree_centrality(graph):
    """Calculate degree centrality for each node in the graph."""
    return sorted(graph.degree, key=lambda x: x[1], reverse=True)

def detect_communities_girvan_newman(G, clone_username, original_username):
    print("Starting community detection using Girvan-Newman")
    comp = community.girvan_newman(G)
    communities = next(comp)  # Get the first partition of communities

    # Initialize the communities for the specified usernames
    clone_community = None
    original_community = None

    # Create a partition dictionary for the community assignment
    partition = {}
    for i, community1 in enumerate(communities):
        for node in community1:
            partition[node] = i  # Assign community ID
            if node == clone_username:
                clone_community = set(community1)
            if node == original_username:
                original_community = set(community1)

    print("Partition created:", partition)

    # Set the node attributes in the graph
    nx.set_node_attributes(G, partition, "community")
    print("Community attributes set for nodes")

    return partition, clone_community, original_community

# Function to calculate modularity
def calculate_modularity(G, partition):
    m = G.number_of_edges()
    modularity = 0
    for community_id in set(partition.values()):
        nodes_in_community = [node for node, comm in partition.items() if comm == community_id]
        subgraph = G.subgraph(nodes_in_community)
        lc = subgraph.number_of_edges()
        dc = sum(G.degree(node) for node in nodes_in_community)
        modularity += (lc / m) - (dc**2 / (4 * m**2)) if m > 0 else 0
    return modularity

# Step 4: Calculate node's local contribution to modularity vitality
def calculate_local_contribution(G, partition, username):
    if username not in partition:
        return None

    community_id = partition[username]
    intra_community_edges = sum(1 for neighbor in G.neighbors(username) if partition[neighbor] == community_id)
    inter_community_edges = G.degree(username) - intra_community_edges
    total_edges = G.number_of_edges()
    local_modularity = (intra_community_edges / total_edges) - \
                       (G.degree(username) ** 2 / (4 * total_edges ** 2)) if total_edges > 0 else 0

    return {
        "intra_community_edges": intra_community_edges,
        "inter_community_edges": inter_community_edges,
        "local_contribution": intra_community_edges - inter_community_edges,
        "local_modularity": local_modularity
    }

def calculate_influence_percentage(G, partition, contribution, modularity):
    total_edges = G.number_of_edges()
    local_modularity = contribution['local_modularity']
    intra_community_edges = contribution['intra_community_edges']
    inter_community_edges = contribution['inter_community_edges']

    weight_local = 0.6
    weight_edges = 0.4
    influence_percentage = (
        (local_modularity / modularity * weight_local) +
        ((intra_community_edges - inter_community_edges) / total_edges * weight_edges)
    ) * 100 if modularity != 0 and total_edges > 0 else 0

    return influence_percentage

# Main function to fetch the graph, detect communities, and display results
def main():
    # Step 1: Fetch the graph from Neo4j
    graph = fetch_graph_from_neo4j()
    
    # Example usernames
    user1 = "catherinemcilkenny"  # Replace with actual username
    user2 = "mcfitzpatrickcat"  # Replace with actual username
    
    # Step 2: Detect communities
    partition, clone_community, original_community = detect_communities_girvan_newman(graph, user1, user2)

    # Step 3: Print communities
    for i in set(partition.values()):
        community_nodes = [node for node, comm in partition.items() if comm == i]
        print(f"Community {i + 1}: {sorted(community_nodes)}")
    
    # Calculate modularity
    modularity = calculate_modularity(graph, partition)
    print(f"Modularity of the detected partition: {modularity}")

    # Calculate local contribution for user2
    contribution_user2 = calculate_local_contribution(graph, partition, user2)
    if contribution_user2:
        print(f"Local contribution for {user2}: {contribution_user2}")

        # Calculate influence percentage for user2
        influence_percentage_user2 = calculate_influence_percentage(graph, partition, contribution_user2, modularity)
        print(f"Influence percentage for {user2}: {influence_percentage_user2:.2f}%")
    else:
        print(f"{user2} not found in partition.")

# Run the main function
if __name__ == "__main__":
    main()
