from neo4j import GraphDatabase
import networkx as nx

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

def expand_community(graph, seed, alpha):
    """Expand a community starting from a seed node."""
    community = set([seed])
    neighbors = set(graph.neighbors(seed))
    
    while neighbors:
        neighbor = neighbors.pop()
        community.add(neighbor)
        
        # Check connections to community members
        connected_to_community = sum(1 for n in graph.neighbors(neighbor) if n in community)
        if connected_to_community / graph.degree(neighbor) >= alpha:
            community.add(neighbor)
            neighbors.update(set(graph.neighbors(neighbor)) - community)
    
    return community

def remove_subset_communities(communities):
    """Remove communities that are subsets of other communities."""
    non_subset_communities = []
    for community in communities:
        if not any(community < other for other in non_subset_communities):
            non_subset_communities.append(community)
    return non_subset_communities

def partition_overlapping_nodes(graph, communities):
    """Partition overlapping nodes among communities."""
    return communities

def detect_communities(graph, alpha=1.0):
    communities = []
    seeds = calculate_degree_centrality(graph)
    
    while seeds:
        seed = seeds.pop(0)[0]  # Get the node name from the degree tuple
        if any(seed in community for community in communities):
            continue
        
        community = expand_community(graph, seed, alpha)
        communities.append(community)
    
    communities = remove_subset_communities(communities)
    communities = partition_overlapping_nodes(graph, communities)
    
    return communities

def get_user_communities(graph, detected_communities, username):
    """Get the communities for a specific username."""
    user_communities = set()
    
    for community in detected_communities:
        if username in community:
            user_communities.add(frozenset(community))  # Using frozenset to make it hashable
            
    return user_communities

def find_intersecting_communities(graph, detected_communities, user1, user2):
    """Find the intersecting communities of two users."""
    user1_communities = get_user_communities(graph, detected_communities, user1)
    user2_communities = get_user_communities(graph, detected_communities, user2)
    
    # Find the intersection of communities
    intersecting_communities = user1_communities.intersection(user2_communities)
    
    # Collect usernames in the intersecting communities
    intersecting_usernames = set()
    for community in intersecting_communities:
        intersecting_usernames.update(community)
    
    return intersecting_usernames

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
    
    # Step 2: Detect communities
    detected_communities = detect_communities(graph, alpha=1.5)

    # Create a partition dictionary for community assignments
    partition = {}
    for i, community in enumerate(detected_communities):
        for username in community:
            partition[username] = i  # Assign community ID

    # Step 3: Print communities
    for i, community in enumerate(detected_communities):
        print(f"Community {i + 1}: {sorted(community)}")
    
    # Example usernames
    user1 = "catherinemcilkenny"  # Replace with actual username
    user2 = "mike-dean-8509a193"  # Replace with actual username
    
    # Step 4: Find intersecting communities
    intersecting_usernames = find_intersecting_communities(graph, detected_communities, user1, user2)
    
    # Step 5: Print intersecting usernames
    print(f"Intersecting communities of {user1} and {user2}: {sorted(intersecting_usernames)}")
    
    # Calculate modularity
    modularity = calculate_modularity(graph, partition)
    print(f"Modularity of the detected partition: {modularity}")

    # Calculate local contribution for user1
    contribution_user2 = calculate_local_contribution(graph, partition, user2)
    if contribution_user2:
        print(f"Local contribution for {user2}: {contribution_user2}")

        # Calculate influence percentage for user1
        influence_percentage_user2 = calculate_influence_percentage(graph, partition, contribution_user2, modularity)
        print(f"Influence percentage for {user2}: {influence_percentage_user2:.2f}%")
    else:
        print(f"{user2} not found in partition.")

# Run the main function
if __name__ == "__main__":
    main()
