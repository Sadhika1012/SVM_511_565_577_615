import random
from neo4j import GraphDatabase

# Define Neo4j connection details
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

# Define connection function
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Define function to create random connections between nodes
def create_random_connections(tx, node_count, start_id, end_id):
    for _ in range(node_count):
        source_node_id = random.randint(start_id, end_id)
        target_node_id = random.randint(start_id, end_id)
        tx.run(
            "MATCH (source:Net), (target:Net) WHERE id(source) = $source_id AND id(target) = $target_id "
            "CREATE (source)-[:CONNECTED_TO]->(target)",
            source_id=source_node_id, target_id=target_node_id
        )

# Establish connections and create random connections between nodes
with connect_to_neo4j(uri, username, password) as driver:
    with driver.session() as session:
        # Define start and end node IDs
        start_node_id = 3946
        end_node_id = 4044
        
        # Generate random connections between nodes
        node_count = 60 # Adjust this to control the number of connections
        session.execute_write(create_random_connections, node_count, start_node_id, end_node_id)
