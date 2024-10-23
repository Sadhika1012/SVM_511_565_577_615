from neo4j import GraphDatabase
from datetime import datetime

# Neo4j connection details
uri = "neo4j+ssc://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

def update_creation_time(uri, user, password, node_id, creation_time):
    """Update the creation_time property of a node in the Neo4j database."""
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database="neo4j") as session:
        query = """
        MATCH (n:N1)
        WHERE id(n) = $node_id
        SET n.creation_time = $creation_time
        """
        session.run(query, node_id=node_id, creation_time=creation_time)
    driver.close()
    print(f"Creation time updated for node with ID {node_id}.")

if __name__ == "__main__":
    # Node ID in Neo4j
    node_id = 1802  # Replace with the actual node ID

    # Update the creation_time to the current date and time
    creation_time = datetime.now().isoformat()

    # Update the Neo4j node with the new creation_time
    update_creation_time(uri, username, password, node_id, creation_time)
