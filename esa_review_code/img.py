from neo4j import GraphDatabase
import base64
    # Neo4j connection details
uri = "neo4j+ssc://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

def convert_image_to_base64(image_path):
    """Convert a JPEG image to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def update_avatar_in_neo4j(uri, user, password, node_id, base64_image):
    """Update the avatar_base64 property of a node in the Neo4j database."""
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database="neo4j") as session:
        query = """
        MATCH (n:N1)
        WHERE id(n) = $node_id
        SET n.avatar_base64 = $base64_image
        """
        session.run(query, node_id=node_id, base64_image=base64_image)
    driver.close()
    print(f"Avatar updated for node with ID {node_id}.")

if __name__ == "__main__":
    # Path to the JPEG image
    image_path = "mod2.jpg"



    # Node ID in Neo4j
    node_id = 1802 # Replace with the actual node ID

    # Convert the image to base64
    base64_image = convert_image_to_base64(image_path)

    # Update the Neo4j node with the base64 encoded image
    update_avatar_in_neo4j(uri, username, password, node_id, base64_image)