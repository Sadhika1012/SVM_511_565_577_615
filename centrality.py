import csv
import random
import networkx as nx
from neo4j import GraphDatabase

# Neo4j connection details
uri = "neo4j+s://108cd070.databases.neo4j.io"
username = "neo4j"
password = "GYxY3TYSE5lT2y6A-tb8NKmX7rLchsh-metIu_jGk3g"

# Function to establish connection to Neo4j
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

def print_centrality_scores(tx):
    result = tx.run(
        "MATCH (p1:Person)-[:CONNECTED_TO]-(p2:Person) "
        "RETURN p1.name AS node_name, collect(p2.name) AS neighbors"
    )
       
    G = nx.Graph()
    for record in result:
        node = record["node_name"]
        neighbors = record["neighbors"]
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Calculate centrality scores
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Central profiles for comparison
    cloned_profile_name = 'mike-dean-8509a193'  #original
    original_profile_name = 'mike-dean-8509a192' #cloned
    
    print("Centrality Scores for Cloned and Original Profiles:")
    print("---------------------")
    
    # Print all centrality measures at once
    for centrality, name in [
        (degree_centrality, "Degree Centrality"),
        (closeness_centrality, "Closeness Centrality"),
        (betweenness_centrality, "Betweenness Centrality"),
        (eigenvector_centrality, "Eigenvector Centrality")
    ]:
        print(f"\n{name}:")
        print(f"{cloned_profile_name}: {centrality.get(cloned_profile_name, 'N/A')}")
        print(f"{original_profile_name}: {centrality.get(original_profile_name, 'N/A')}")
    
    # Find common neighbors
    common_neighbors = list(nx.common_neighbors(G, cloned_profile_name, original_profile_name))
    
    print("\nCommon Neighbors:")
    if common_neighbors:
        for neighbor in common_neighbors:
            print(neighbor)
    else:
        print("No common neighbors found.")

# Function to create profiles and relationships in the Neo4j database
def create_profiles_and_relationships(tx):
    # Create the original and cloned profiles
    tx.run(
        """
        MERGE (p1:Person {name: 'mike-dean-8509a192'})
        MERGE (p2:Person {name: 'mike-dean-8509a193'})
        MERGE (p3:Person {name: 'margot'})
        MERGE (p4:Person {name: 'james'})
        MERGE (p5:Person {name: 'sophie'})
        
        // Avoid creating duplicate connections by merging only once in one direction
        MERGE (p1)-[:CONNECTED_TO]->(p3)
        MERGE (p1)-[:CONNECTED_TO]->(p4)
        MERGE (p1)-[:CONNECTED_TO]->(p5)
        MERGE (p2)-[:CONNECTED_TO]->(p3)
        MERGE (p2)-[:CONNECTED_TO]->(p4)
        """
    )
    print("Profiles and relationships created.")

# Function to notify all neighbors of the cloned profile
def notify_all_neighbors_of_clone(tx, cloned_profile_name):
    result = tx.run(
        """
        MATCH (p1:Person {name: $cloned_profile_name})-[:CONNECTED_TO]-(p2:Person)
        RETURN DISTINCT p2.name AS neighbor_name
        """,
        cloned_profile_name=cloned_profile_name
    )
    neighbors = [record["neighbor_name"] for record in result]
    
    if neighbors:
        print(f"Notifying all neighbors of the cloned profile {cloned_profile_name}:")
        for neighbor in neighbors:
            tx.run(
                """
                MATCH (p1:Person {name: $cloned_profile_name}), (p2:Person {name: $neighbor})
                MERGE (p1)-[:CLONE_NEIGHBOR]->(p2)
                """,
                cloned_profile_name=cloned_profile_name, neighbor=neighbor
            )
            print(f"Notified {neighbor} about the cloned profile {cloned_profile_name}.")
    else:
        print(f"No neighbors found to notify for the cloned profile {cloned_profile_name}.")
if __name__ == "__main__":
    cloned_profile_name = 'mike-dean-8509a193'  # Cloned profile name
    original_profile_name = 'mike-dean-8509a192'  # Original profile name


    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.write_transaction(create_profiles_and_relationships)

  
    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.execute_read(print_centrality_scores)


    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.write_transaction(notify_all_neighbors_of_clone, cloned_profile_name)
