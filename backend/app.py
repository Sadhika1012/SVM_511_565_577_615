from flask import Flask, request, jsonify, session as flask_session, redirect, url_for
from flask_cors import CORS
from neo4j import GraphDatabase
import uuid
import bcrypt
import secrets


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
app.secret_key = secrets.token_hex(16)

#defining neo4j connections
uri = "neo4j+ssc://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"
driver = GraphDatabase.driver(uri, auth=("neo4j", password))

# Define connection function
def connect_to_neo4j():
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to fetch graph data from Neo4j
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
        # print("Nodes:")
        # print(nodes)
        # print("Edges:")
        # print(edges)

        return {'nodes': nodes, 'edges': edges}
    
def add_hashed_password(session, element_id, password):
    # Hash the password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # Add or update the password attribute on the node
    session.run(
        "MATCH (n) WHERE id(n) = $element_id "
        "SET n.password = $hashed_password",
        element_id=element_id, hashed_password=hashed
    )
    print(f"Password for node {element_id} has been set.")

# def print_node_attributes(session, element_id):
#     """Fetch and print the attributes of a node in the Neo4j database."""
#     result = session.run(
#         "MATCH (n) WHERE id(n) = $element_id RETURN n",
#         element_id=element_id
#     )

#     # Iterate over the result
#     node = result.single()
#     if node:
#         node_properties = node["n"].items()
#         print(f"Attributes for node with element_id {element_id}:")
#         for key, value in node_properties:
#             print(f"{key}: {value}")
#     else:
#         print(f"No node found with element_id {element_id}")

# def delete_password(session, element_id):
#     # Remove the password attribute from the node
#     session.run(
#         "MATCH (n) WHERE id(n) = $id "
#         "REMOVE n.password",
#         element_id=element_id
#     )
#     print(f"Password for node {element_id} has been deleted.")
# def find_user(tx, name):
#     query = "MATCH (u:User {name: $name}) RETURN u"
#     result = tx.run(query, name=name)
#     return result.single()

# def authenticate_user(neo4j_session, name, password):
#     print("Attempting to authenticate user...")
#     user = neo4j_session.execute_read(find_user, name)
    
#     if user:
#         user_data = user['u']
#         password_bytes = password.encode('utf-8')
#         stored_password_bytes = user_data['password'].encode('utf-8')

#         # Display the stored hashed password for debugging (ensure it's not exposed in production)
#         print("Stored hashed password:", stored_password_bytes)

#         # Verify the entered password against the stored hashed password
#         if bcrypt.checkpw(password_bytes, stored_password_bytes):
#             print("Authentication successful.")
#             return {
#                 'name': user_data['name'],
#                 'city': user_data['city'],
#                 'position': user_data['position']
#             }
#         else:
#             print("Authentication failed: Password does not match.")
#     else:
#         print("Authentication failed: User not found.")
    
#     return None


@app.route('/graph')
def get_graph_data():
    graph_data = fetch_graph_data()
    return jsonify(graph_data)

# @app.route('/')
# def index():
#     if 'user' in flask_session:
#         return redirect(url_for('profile'))
#     return redirect(url_for('login'))

# @app.route('/profile')
# def profile():
#     if 'user' not in flask_session:
#         print("user not in flask idk why...")
#         return redirect(url_for('login'))
#     user = flask_session['user']
#     return jsonify(user)

# @app.route('/logout')
# def logout():
#     flask_session.pop('user', None)
#     return redirect(url_for('login'))

@app.route('/login', methods=['POST','GET'])
def login():
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json sadhika"}), 415
    data = request.get_json()
   
    if not data or 'name' not in data or 'password' not in data:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400

    name = data.get('name')
    password = data.get('password')
    
    with connect_to_neo4j().session() as session:
        user = session.run("MATCH (u:Network {name: $name}) RETURN u", name=name).single()

        if user:
            user_data = user['u']
            stored_password_hash = user_data['password'].encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                flask_session['user'] = {
                    'name': user_data['name'],
                    'city': user_data['city'],
                    'position': user_data['position']
                }
                print("successful!!")
                return jsonify({"success": True, "user": flask_session['user']})#getting returned to react
        
    return jsonify({"success": False, "message": "Invalid name or password"}), 401


@app.route('/api/addpwd', methods=['GET', 'POST'])
def add_pwd_db():
    element_id = int(input("Enter the element ID of the node: "))   # Replace with the actual internal ID of the node
    password_to_add =input("Enter the password: ")
    
    # Connect to Neo4j and execute the function
    driver = connect_to_neo4j()
    with driver.session() as session:
        add_hashed_password(session, element_id, password_to_add)

# @app.route('/api/check-auth', methods=['GET'])
# def check_auth():
#     if 'user' in flask_session:
#         return jsonify({"authenticated": True})
#     return jsonify({"authenticated": False})  



if __name__ == '__main__':
    app.run(debug=True)
