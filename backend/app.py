import hashlib
from flask import Flask, make_response, request, jsonify, session as flask_session, redirect, url_for
from flask_cors import CORS
from neo4j import GraphDatabase
import uuid
import bcrypt
import secrets
import pandas as pd
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


app = Flask(__name__)
# CORS configuration
CORS(app, supports_credentials=True)
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
        
        result = session.run("MATCH (n1:N1)-[r]->(n2:N1) RETURN n1, r, n2")
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

@app.route('/logout', methods=['POST'])
def logout():
    flask_session.pop('user', None)  # Remove the user from the session
    response = make_response(jsonify({"success": True, "message": "Logged out successfully"}))
    return response



# Define function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json"}), 415

    data = request.get_json()

    if not data or 'name' not in data or 'password' not in data:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400

    name = data.get('name')
    password = data.get('password')
    
    with connect_to_neo4j().session() as session:
        user = session.run("MATCH (u:N1 {name: $name}) RETURN u", name=name).single()

        if user:
            user_data = user['u']
            stored_password_hash = user_data.get('password')

            # Hash the input password
            hashed_input_password = hash_password(password)

            if hashed_input_password == stored_password_hash:
                flask_session['user'] = {
                    'username':user_data['username'],
                    'name': user_data['name'],
                    'city': user_data['city'],
                    'position': user_data['position'],
                    'about': user_data.get('about', ''),
                    'education': user_data.get('education', '')
                }
                return jsonify({"success": True, "user": flask_session['user']})

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

# Load model from HuggingFace Hub once
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Function to concatenate profile values into a single string, considering only non-NaN attributes
def concatenate_profile(profile, valid_attributes):
    return " ".join([str(profile[attr]) for attr in valid_attributes if pd.notna(profile[attr])])

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to compute cosine similarity between a source profile and a list of target profiles
def compute_similarity_scores(source_profile, target_profiles):
    similarity_scores = []
    for target_profile in target_profiles:
        valid_attributes = [attr for attr in target_profile.keys() if pd.notna(target_profile[attr])]
       
        
        modified_source_sentence = concatenate_profile(source_profile, valid_attributes)
        modified_target_sentence = concatenate_profile(target_profile, valid_attributes)

        encoded_input = tokenizer([modified_source_sentence, modified_target_sentence], padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        cosine_similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0)).item()
        levenshtein_similarity = Levenshtein.ratio(modified_source_sentence, modified_target_sentence)
        combined_similarity = (cosine_similarity * 0.85 + levenshtein_similarity * 0.15)

        similarity_scores.append(combined_similarity)

    return similarity_scores

@app.route('/find-clones', methods=['POST'])
def find_clones():
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json"}), 415
    
    data = request.get_json()
    print('Received data:', data)  # Print received data for debugging
    
    if not data:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400
    
    # Clean user_profile to remove or replace NaN values
    cleaned_profile = {k: (v if pd.notna(v) else '') for k, v in data.items()}
    
    # Read profiles from CSV
    csv_file_path = 'newdataset3.csv'
    df = pd.read_csv(csv_file_path)
    
    # Replace NaN values with empty strings in the DataFrame
    df = df.fillna('')
    
    # Source profile
    source_profile = cleaned_profile
    print('Source Profile', source_profile)
    
    # List of target profiles (excluding the source profile)
    target_profiles = df.to_dict(orient='records')
    
    # Compute similarity scores
    similarity_scores = compute_similarity_scores(source_profile, target_profiles)
    
    # Include scores with profiles
    result = [{"profile": profile, "score": score} for profile, score in zip(target_profiles, similarity_scores)]
    
    return jsonify({"success": True, "result": result})

if __name__ == '__main__':
    app.run(debug=True)
