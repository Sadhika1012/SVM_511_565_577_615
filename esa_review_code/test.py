import networkx as nx
import igraph as ig
from cdlib import algorithms
from neo4j import GraphDatabase
import leidenalg as la  # Direct import of leidenalg for detailed control

# Neo4j connection settings
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"
driver = GraphDatabase.driver(uri, auth=(username, password))

def fetch_graph_from_neo4j():
    edge_query = """
    MATCH (n:Net)-[r]->(m:Net)
    RETURN n.username AS source, m.username AS target, type(r) AS relationship
    """
    node_query = """
    MATCH (n:Net)
    RETURN n.username AS username
    """
    print("Starting graph creation based on usernames...")

    with driver.session() as session:
        edge_result = session.run(edge_query)
        edges = [(record['source'], record['target']) for record in edge_result]
        print(f"Number of edges fetched: {len(edges)}")

        node_result = session.run(node_query)
        nodes = [record['username'] for record in node_result]
        print(f"Number of nodes fetched: {len(nodes)}")

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def convert_nx_to_igraph(G_nx):
    edges = list(G_nx.edges())
    nodes = list(G_nx.nodes())

    G_ig = ig.Graph()
    G_ig.add_vertices(len(nodes))
    
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    edges_for_igraph = [(node_to_index[source], node_to_index[target]) for source, target in edges]
    
    G_ig.add_edges(edges_for_igraph)
    G_ig.vs["username"] = nodes
    
    # Add default attributes to edges
    G_ig.es["weight"] = [1] * len(G_ig.es)  # Default weight
    G_ig.es["relationship"] = ["default"] * len(G_ig.es)  # Relationship type
    
    print(f"iGraph conversion complete. Graph has {len(G_ig.vs)} vertices and {len(G_ig.es)} edges with default weights and relationships.")
    return G_ig

def detect_communities(G_ig, k):
    """Detect communities using Leiden directly for more control."""
    try:
        print(f"Attempting community detection with k={k}")
        
        # Use leidenalg directly
        partition = la.find_partition(G_ig, la.CPMVertexPartition, resolution_parameter=k)
        communities_with_usernames = []
        
        for community in partition:
            community_usernames = [G_ig.vs[node]["username"] for node in community]
            communities_with_usernames.append(community_usernames)
        
        print("Successfully detected communities using CPM")
        return communities_with_usernames

    except Exception as e:
        print(f"Error during direct CPM community detection: {str(e)}")
        
        try:
            # Attempt Louvain as an alternative
            communities = algorithms.louvain(G_ig)
            communities_with_usernames = []
            
            for community in communities.communities:
                community_usernames = [G_ig.vs[node]["username"] for node in community]
                communities_with_usernames.append(community_usernames)
                
            print("Successfully detected communities using Louvain")
            return communities_with_usernames
            
        except Exception as e:
            print(f"Error during Louvain detection: {str(e)}")
            return None
        

def find_overlapping_communities(G_ig, k, username1, username2):
    """Detect communities and find overlapping communities for two specific usernames."""
    communities = detect_communities(G_ig, k)
    
    if not communities:
        print("No communities were detected.")
        return None
    
    overlapping_communities = []
    
    for community in communities:
        if username1 in community and username2 in community:
            overlapping_communities.append(community)
    
    if overlapping_communities:
        print(f"\nOverlapping communities for {username1} and {username2}:")
        for i, community in enumerate(overlapping_communities):
            print(f"Community {i + 1} (size {len(community)}): {community}")
    else:
        print(f"No overlapping communities found for {username1} and {username2}.")
    
    return overlapping_communities

def extract_user_subgraph(G, username1, radius):
    subgraph = nx.ego_graph(G, username1, radius=radius)
    print(f"Sub-community around {username} within {radius} degrees of separation:")
    print(subgraph.nodes())
    return subgraph

# Main execution
# Main execution
try:
    G = fetch_graph_from_neo4j()
    G_igraph = convert_nx_to_igraph(G)
    k_value = 0.4  # Adjusted for CPM detection, you can tune it as needed
    username1 = "mcfitzpatrickcat"  # Replace with the actual username
    username2 = "shakeel-ahmed-59311b114"  # Replace with the actual username
    radius=2
    
    overlapping_communities = find_overlapping_communities(G_igraph, k_value, username1, username2)
    subgraph=extract_user_subgraph(G,username1,radius)
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    driver.close()

