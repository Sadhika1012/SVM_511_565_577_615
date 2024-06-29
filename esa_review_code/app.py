from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
import json
import csv
import uuid
from datetime import datetime

app = Flask(__name__)

# Define Neo4j connection details
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"

# Define connection function
def connect_to_neo4j():
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to fetch graph data from Neo4j
def fetch_graph_data():
    with connect_to_neo4j().session() as session:
        result = session.run("MATCH (n1:Network)-[r]->(n2:Network) RETURN n1, r, n2")
        nodes = []
        edges = []
        for record in result:
            source_node = record['n1']
            target_node = record['n2']
            edge = {'id': str(uuid.uuid4()), 'from': source_node.element_id, 'to': target_node.element_id}
            edges.append(edge)
            
            source_properties = {'id': source_node.element_id, 'label': source_node['name']}
            target_properties = {'id': target_node.element_id, 'label': target_node['name']}
            
            if source_properties not in nodes:
                nodes.append(source_properties)
            if target_properties not in nodes:
                nodes.append(target_properties)
        
        return {'nodes': nodes, 'edges': edges}

# Define route to visualize the graph
@app.route('/')
def visualize_graph():
    graph_data = fetch_graph_data()
    return render_template('visualise.html', graph_data=json.dumps(graph_data))

# Define route to add a new node and relationships (GET method)
@app.route('/add_node', methods=['GET'])
def get_add_node():
    # Fetch existing nodes from Neo4j to display in the form
    with connect_to_neo4j().session() as session:
        result = session.run("MATCH (n:Network) RETURN ID(n) AS node_id, n.name AS name")
        existing_nodes = [{'node_id': record['node_id'], 'name': record['name']} for record in result]

    return render_template('add_node.html', existing_nodes=existing_nodes)

# Define route to add a new node and relationships (POST method)
@app.route('/add_node', methods=['POST'])
def post_add_node():
    name = request.form['name']
    city = request.form['city']
    position = request.form['position']

    with connect_to_neo4j().session() as session:
        # Create node in Neo4j graph with creation time
        result = session.run(
            "CREATE (n:Network {name: $name, city: $city, position: $position, creation_time: $creation_time}) RETURN ID(n) AS node_id",
            name=name, city=city, position=position, creation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ).single()

        # Check if the node was created successfully
        if result is not None:
            new_node_id = result['node_id']

            # Allow the user to choose existing nodes to connect to
            selected_nodes = request.form.getlist('selected_nodes')

            # Create 'connected_to' relationships between the new node and selected nodes
            for existing_node_id in selected_nodes:
                session.run(
                    "MATCH (n1:Network), (n2:Network) WHERE ID(n1) = $new_node_id AND ID(n2) = $existing_node_id "
                    "CREATE (n1)-[:CONNECTED_TO]->(n2)",
                    new_node_id=new_node_id, existing_node_id=int(existing_node_id)
                )

            return 'Node added successfully.'
        else:
            return 'Node creation failed.'

# Define route to get existing nodes
@app.route('/get_existing_nodes', methods=['GET'])
def get_existing_nodes():
    with connect_to_neo4j().session() as session:
        result = session.run("MATCH (n:Network) RETURN ID(n) AS node_id, n.name AS name")
        existing_nodes = [{'node_id': record['node_id'], 'name': record['name']} for record in result]
        return jsonify(existing_nodes)

# Function to get node information based on ID
def get_node_info(node_id):
    with connect_to_neo4j().session() as session:
        result = session.run("MATCH (n:Network) WHERE id(n) = $node_id RETURN n.name AS name, n.city AS city, n.position AS position, n.creation_time AS creation_time", node_id=node_id)
        return result.single()

# Flask route to retrieve node information
@app.route('/get_node_info/<int:node_id>', methods=['GET'])
def get_node_information(node_id):
    node_info = get_node_info(node_id)
    if node_info:
        return jsonify({'name': node_info['name'], 'city': node_info['city'], 'position': node_info['position'], 'creation_time': node_info['creation_time']})
    else:
        return jsonify({'error': 'Node not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
