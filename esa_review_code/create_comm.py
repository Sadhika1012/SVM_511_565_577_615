from neo4j import GraphDatabase
import networkx as nx
from networkx.algorithms.community import girvan_newman
import random

# Updated Neo4j connection details
uri = "neo4j+ssc://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

# Fetch graph data from Neo4j specific to network 'N4'
def get_graph_data():
    query = """
    MATCH (n:Net)-[r]->(m:Net)
    RETURN id(n) AS source, id(m) AS target, type(r) AS CONNECTED_TO
    """
    with driver.session() as session:
        result = session.run(query)
        # Filter out any edges with None values
        edges = [(record['source'], record['target']) for record in result if record['source'] and record['target']]
    print(f"Fetched {len(edges)} edges from the database.")
    return edges

# Convert Neo4j data into a NetworkX graph
def build_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# Detect communities using Girvan-Newman algorithm
def detect_communities(graph, target_communities=3):
    communities_generator = girvan_newman(graph)
    for i, communities in enumerate(communities_generator):
        print(f"Iteration {i+1}: Detected {len(communities)} communities.")
        if len(communities) <= target_communities:
            break
    # Convert communities from list of sets to a dictionary mapping node to community ID
    partition = {node: i for i, community in enumerate(communities) for node in community}
    return partition

# Function to add edges between disconnected nodes in the same community
def connect_communities(graph, partition, target_communities=3):
    # Detect the number of communities
    community_sets = {}
    for node, community in partition.items():
        if community not in community_sets:
            community_sets[community] = []
        community_sets[community].append(node)
    
    # Sort communities by size (largest to smallest)
    sorted_communities = sorted(community_sets.values(), key=len, reverse=True)
    
    # If we already have the target number of communities, return without changes
    if len(sorted_communities) == target_communities:
        print(f"Already have {target_communities} communities.")
        return graph

    # If we have more than the target number of communities, we merge smaller ones
    while len(sorted_communities) > target_communities:
        # Get two smallest communities
        community1 = sorted_communities.pop()
        community2 = sorted_communities.pop()
        
        # Add a random connection between them
        node1 = random.choice(community1)
        node2 = random.choice(community2)
        graph.add_edge(node1, node2)
        print(f"Added edge between {node1} and {node2} to merge communities.")
        
        # Merge the two communities into one
        new_community = community1 + community2
        sorted_communities.append(new_community)

    return graph

# Function to update the Neo4j graph with new connections
def update_graph_in_neo4j(graph, new_edges):
    with driver.session() as session:
        for edge in new_edges:
            source, target = edge
            query = f"""
            MATCH (a:Net {{id: '{source}'}}), (b:Net {{id: '{target}'}})
            MERGE (a)-[:NEW_CONNECTION]->(b)
            """
            session.run(query)
    print(f"Updated {len(new_edges)} new edges in the database.")

# Main execution
try:
    edges = get_graph_data()
    if not edges:
        print("No edges fetched from the database.")
    else:
        graph = build_graph(edges)
        partition = detect_communities(graph, target_communities=3)

        # Add connections to reduce the number of communities to 3
        manipulated_graph = connect_communities(graph, partition, target_communities=3)

        # Get new edges added to the graph
        new_edges = [(u, v) for u, v in manipulated_graph.edges() if (u, v) not in edges]

        # Update Neo4j with the new edges
        if new_edges:
            update_graph_in_neo4j(manipulated_graph, new_edges)
        else:
            print("No new edges to update in the database.")
finally:
    # Close the Neo4j connection
    driver.close()
