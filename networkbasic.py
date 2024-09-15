from neo4j import GraphDatabase

# Neo4j connection details
uri = "neo4j+s://acc005ca.databases.neo4j.io"
username = "neo4j"
password = "div_QO7aDVKf_CadTZnGSu2DwSzCrtvipR_5qwqKUMo"

# Function to establish connection to Neo4j
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to find nodes that have 'margot' as a neighbor and print details
def find_and_print_margot_neighbors(tx, cloned_profile_name):
    result = tx.run(
        "MATCH (p1:Person {name: $cloned_profile_name})-[:CONNECTED_TO]-(p2:Person) "
        "RETURN p2.name AS neighbor_name",
        cloned_profile_name=cloned_profile_name
    )
    neighbors = [record["neighbor_name"] for record in result]
    if neighbors:
        print(f"Nodes with {cloned_profile_name} as neighbor:")
        print("-----------------------------------")
        for neighbor in neighbors:
            print(f"Node: {neighbor}")
            # Print all neighbors of this node
            result = tx.run(
                "MATCH (p1:Person {name: $neighbor})-[:CONNECTED_TO]-(p2:Person) "
                "RETURN p2.name AS neighbor_name",
                neighbor=neighbor
            )
            node_neighbors = [record["neighbor_name"] for record in result]
            print(f"Neighbors: {', '.join(node_neighbors)}")
            print("-----------------------------------")
    else:
        print(f"No nodes found with {cloned_profile_name} as neighbor.")

# Function to notify nodes about a cloned profile
def notify_cloned_profile(tx, cloned_profile_name):
    result = tx.run(
        "MATCH (p1:Person {name: $cloned_profile_name})-[:CONNECTED_TO]-(p2:Person) "
        "RETURN p2.name AS neighbor_name",
        cloned_profile_name=cloned_profile_name
    )
    neighbors = [record["neighbor_name"] for record in result]
    for neighbor in neighbors:
        tx.run(
            "MATCH (p1:Person {name: $cloned_profile_name}), (p2:Person {name: $neighbor}) "
            "MERGE (p1)-[:CLONE_NEIGHBOR]->(p2)",
            cloned_profile_name=cloned_profile_name, neighbor=neighbor
        )
        print(f"Notified {neighbor} that {cloned_profile_name} is a cloned profile.")

# Function to remove connections between cloned profile and its neighbors
def remove_cloned_profile_connections(tx, cloned_profile_name):
    # Remove all relationships of the cloned profile
    tx.run(
        "MATCH (p:Person {name: $cloned_profile_name})-[r]-() "
        "DELETE r",
        cloned_profile_name=cloned_profile_name
    )
    # Remove the cloned profile node itself
    tx.run(
        "MATCH (p:Person {name: $cloned_profile_name}) "
        "DELETE p",
        cloned_profile_name=cloned_profile_name
    )

# Function to print all nodes and their neighbors
def print_all_nodes_and_neighbors(tx):
    result = tx.run(
        "MATCH (p1:Person)-[:CONNECTED_TO]-(p2:Person) "
        "RETURN p1.name AS node_name, collect(p2.name) AS neighbors"
    )
    nodes = {record["node_name"]: record["neighbors"] for record in result}
    print("All nodes and their neighbors:")
    print("-----------------------------------")
    for node, neighbors in nodes.items():
        print(f"Node: {node}")
        print(f"Neighbors: {', '.join(neighbors)}")
        print("-----------------------------------")

# Entry point to update Neo4j and execute steps
if __name__ == "__main__":
    cloned_profile_name = 'mike-dean-8509a193'  # Replace with the cloned profile name

    # Step 1: Print nodes that have 'margot' as a neighbor
    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.read_transaction(find_and_print_margot_neighbors, cloned_profile_name)
    
    # Step 2: Notify nodes about 'margot' being a cloned profile
    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.write_transaction(notify_cloned_profile, cloned_profile_name)

    # Step 3: Remove connections between 'margot' and its neighbors and delete the node
    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.write_transaction(remove_cloned_profile_connections, cloned_profile_name)

    # Step 4: Print all nodes and their neighbors to indicate 'margot' has been removed
    with connect_to_neo4j(uri, username, password) as driver:
        with driver.session() as session:
            session.read_transaction(print_all_nodes_and_neighbors)
