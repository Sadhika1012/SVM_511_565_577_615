import csv
import random
from random import randint
from neo4j import GraphDatabase
from datetime import datetime

# Define Neo4j connection details
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

# Define connection function
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Define function to create nodes
def create_node(tx, name, city, position, creation_time):
    tx.run(
        "CREATE (n:Network {name: $name, city: $city, position: $position, creation_time: $creation_time})",
        name=name, city=city, position=position, creation_time=creation_time
    )

# Define function to create random connections between nodes
def create_random_connections(tx, node_count):
    for _ in range(node_count):
        source_node_id = randint(1, node_count)
        target_node_id = randint(1, node_count)
        tx.run(
            "MATCH (source:Network), (target:Network) WHERE id(source) = $source_id AND id(target) = $target_id "
            "CREATE (source)-[:CONNECTED_TO]->(target)",
            source_id=source_node_id, target_id=target_node_id
        )

# Establish connections
with connect_to_neo4j(uri, username, password) as driver:
    with driver.session() as session:
        # Read data from CSV and create nodes
        with open('test.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                city = row['city']
                position = row['position']
                creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session.write_transaction(create_node, name, city, position, creation_time)

        