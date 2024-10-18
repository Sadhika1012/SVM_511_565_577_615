import hashlib
from flask import Flask, make_response, request, jsonify, session as flask_session, redirect, url_for
from flask_cors import CORS
from neo4j import GraphDatabase
import base64
import uuid
import bcrypt
import secrets
import pandas as pd
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import requests
import jellyfish
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
import torch
import torch.nn.functional as F
import requests
from deepface import DeepFace
import base64
import io
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim



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
    """
    Fetches graph data from the Neo4j database by querying nodes and edges.

    Returns:
        dict: A dictionary containing nodes and edges extracted from the graph.
    """
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
    """
    Hashes the provided password using SHA-256.

    Args:
        password (str): The password to hash.

    Returns:
        str: The hashed password in hexadecimal format.
    """
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
                    'education': user_data.get('education', ''),
                    'avatar': user_data.get('avatar_base64', '')
                }
                print("Avatar Base64:", user_data.get('avatar_base64', ''))
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
    """
    Concatenates non-NaN attributes of a profile into a single string.

    Args:
        profile (dict): The profile dictionary with attribute names and values.
        valid_attributes (list): A list of valid attribute names to include in the concatenation.

    Returns:
        str: A concatenated string of the profile's non-NaN attributes.
    """
    return " ".join([str(profile[attr]) for attr in valid_attributes if pd.notna(profile[attr])])

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling on the token embeddings considering the attention mask.

    Args:
        model_output (torch.Tensor): The output from the transformer model (token embeddings).
        attention_mask (torch.Tensor): The attention mask used during tokenization.

    Returns:
        torch.Tensor: The pooled sentence embedding.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to compute cosine similarity between a source profile and a list of target profiles
# def compute_similarity_scores(source_profile, target_profiles):
#     """
#     Computes similarity scores between a source profile and a list of target profiles using 
#     cosine similarity and Levenshtein ratio.

#     Args:
#         source_profile (dict): The profile data of the source user.
#         target_profiles (list of dict): A list of target profiles for comparison.

#     Returns:
#         list: A list of similarity scores for each target profile.
#     """
#     similarity_scores = []
#     for target_profile in target_profiles:
#         valid_attributes = [attr for attr in target_profile.keys() if pd.notna(target_profile[attr])]
#         if 'username' in valid_attributes:
#             valid_attributes.remove('username')
       
        
#         modified_source_sentence = concatenate_profile(source_profile, valid_attributes)
#         modified_target_sentence = concatenate_profile(target_profile, valid_attributes)

#         encoded_input = tokenizer([modified_source_sentence, modified_target_sentence], padding=True, truncation=True, return_tensors='pt')

#         with torch.no_grad():
#             model_output = model(**encoded_input)

#         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#         sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#         cosine_similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0)).item()
#         levenshtein_similarity = Levenshtein.ratio(modified_source_sentence, modified_target_sentence)
#         combined_similarity = (cosine_similarity * 0.85 + levenshtein_similarity * 0.15)

#         similarity_scores.append(combined_similarity)

#     return similarity_scores



# Define a simple QWERTY keyboard layout
keyboard_layout = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    'a': (1, 0.5), 's': (1, 1.5), 'd': (1, 2.5), 'f': (1, 3.5), 'g': (1, 4.5),
    'h': (1, 5.5), 'j': (1, 6.5), 'k': (1, 7.5), 'l': (1, 8.5),
    'z': (2, 1), 'x': (2, 2), 'c': (2, 3), 'v': (2, 4), 'b': (2, 5),
    'n': (2, 6), 'm': (2, 7)
}

def keyboard_distance(s1, s2):
    distance = 0
    len_diff = abs(len(s1) - len(s2))
    distance += len_diff * 2  # Penalize for differing lengths

    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        char1 = s1[i].lower()
        char2 = s2[i].lower()
        if char1 in keyboard_layout and char2 in keyboard_layout:
            pos1 = keyboard_layout[char1]
            pos2 = keyboard_layout[char2]
            # Euclidean distance
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            distance += dist
        else:
            # If character not in layout, assign a default distance
            distance += 2
    return distance

# Function to compute cosine similarity between a source profile and a list of target profiles
def compute_similarity_scores(source_profile, target_profiles):
    similarity_scores = []
    
    for target_profile in target_profiles:
        # valid_attributes = [attr for attr in target_profile.keys() if pd.notna(target_profile[attr])]
        valid_attributes = [attr for attr in target_profile.keys() if pd.notna(target_profile[attr])]
        if 'username' in valid_attributes:
            valid_attributes.remove('username')
        print(valid_attributes)
       
        modified_source_sentence = concatenate_profile(source_profile, valid_attributes)
        modified_target_sentence = concatenate_profile(target_profile, valid_attributes)

        # Compute Cosine Similarity
        encoded_input = tokenizer([modified_source_sentence, modified_target_sentence], padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        cosine_similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0)).item()

        # Compute Jaro-Winkler Similarity
        jaro_winkler_similarity = jellyfish.jaro_winkler_similarity(modified_source_sentence, modified_target_sentence)
        print("jaro-winkler similarity is: ", jaro_winkler_similarity)

        # Compute Damerau-Levenshtein Distance and normalize it
        damerau_levenshtein_distance = jellyfish.damerau_levenshtein_distance(modified_source_sentence, modified_target_sentence)
        max_len = max(len(modified_source_sentence), len(modified_target_sentence))
        damerau_levenshtein_similarity = 1 - damerau_levenshtein_distance / max_len if max_len > 0 else 0

        # Compute Keyboard Distance and normalize it
        keyboard_dist = keyboard_distance(modified_source_sentence, modified_target_sentence)
        # Assuming maximum possible keyboard distance as 20 for normalization (adjust as needed)
        keyboard_similarity = 1 - min(keyboard_dist / 20, 1)

        # Compute Jaccard Index
        # Tokenize the sentences
        vectorizer = CountVectorizer().fit([modified_source_sentence, modified_target_sentence])
        vec1 = vectorizer.transform([modified_source_sentence]).toarray()[0]
        vec2 = vectorizer.transform([modified_target_sentence]).toarray()[0]
        if np.count_nonzero(vec1) == 0 or np.count_nonzero(vec2) == 0:
            jaccard_sim = 0
        else:
            jaccard_sim = jaccard_score(vec1, vec2, average='micro')

        # Aggregate similarities with weights
        # Adjust weights as per the importance of each metric
        combined_similarity = (
            cosine_similarity * 0.4 +
            jaro_winkler_similarity * 0.2 +
            damerau_levenshtein_similarity * 0.2 +
            keyboard_similarity * 0.1 +
            jaccard_sim * 0.1
        )

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

# Endpoint to store flagged clone

flagged_clones = []

# Endpoint to store flagged clone
@app.route('/flagged-clones', methods=['POST'])
def flag_clone():
    data = request.get_json()
    print(data)
    original_username = data.get('originalUsername')
    flagged_username = data.get('flaggedUsername')

    # Add the flagged clone to storage
    flagged_clones.append({
        'originalUsername': original_username,
        'flaggedUsername': flagged_username
    })

    return jsonify({'message': 'Clone flagged successfully'}), 200

# Endpoint to retrieve all flagged clones
@app.route('/flagged-clones', methods=['GET'])
def get_flagged_clones():
    return jsonify(flagged_clones), 200

# Endpoint to delete a flagged clone by username
@app.route('/flagged-clones/<flagged_username>', methods=['DELETE'])
def delete_flagged_clone(flagged_username):
    global flagged_clones
    flagged_clones = [clone for clone in flagged_clones if clone['flaggedUsername'] != flagged_username]
    
    return jsonify({'message': 'Clone deleted successfully'}), 200

#----------------add profile pic----------------

# Helper function to update user with base64 avatar
def update_user_avatar(tx, username, avatar_base64):
    query = """
    MATCH (u:N1 {username: $username})
    SET u.avatar_base64 = $avatar_base64
    """
    #print("This is base64",avatar_base64)
    tx.run(query, username=username, avatar_base64=avatar_base64)

@app.route('/upload_csv', methods=['POST','GET'])
def upload_csv():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('newdataset_images_reduced.csv')
    print("inside upload_csv...")

    # Iterate through each row in the CSV
    with driver.session() as session:
        for index, row in df.iterrows():
            username = row['username']
            avatar_url = row['avatar']  # Get the avatar URL from the CSV
            
            print(f"Processing user: {username}, avatar URL: {avatar_url}")

            # If avatar_url is null or empty, assign an empty string
            if pd.isna(avatar_url) or avatar_url.strip() == '':
                print(f"Assigning empty string for user {username} due to null or empty avatar URL.")
                image_base64 = ''  # Assign empty string
            else:
                try:
                    # Download the image from the avatar URL
                    response = requests.get(avatar_url)
                    
                    if response.status_code == 200:
                        # Convert the image to base64
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        #print("This is base64",image_base64)
                    else:
                        print(f"Failed to download image for {username}, status code: {response.status_code}")
                        image_base64 = ''  # Assign empty string on failure

                except Exception as e:
                    print(f"Error downloading image for user {username}: {str(e)}")
                    image_base64 = ''  # Assign empty string on error

            # Update the user with the base64-encoded image in Neo4j
            session.write_transaction(update_user_avatar, username, image_base64)

    return jsonify({'message': 'CSV processed successfully'}), 200

#----------------profile pic comparison----------------
def get_all_profiles():
    # Retrieve all profiles with avatar_base64 from Neo4j
    query = "MATCH (n:N1) RETURN n.username as username, n.avatar_base64 as avatar_base64"
    with driver.session() as session:
        result = session.run(query)
        return [{"username": record["username"], "avatar_base64": record["avatar_base64"]} for record in result]


def decode_base64_image(base64_str):
    # Decode base64 string to an image
    image_data = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image

@app.route('/compare-avatars', methods=['POST'])
def compare_avatars():
    current_profile = request.json.get('current_profile')
    if not current_profile or not current_profile.get('avatar_base64'):
        return jsonify({"error": "Current profile's avatar_base64 is missing"}), 400

    current_avatar_base64 = current_profile['avatar_base64']
    if not current_avatar_base64 or current_avatar_base64.strip() == "":
        return jsonify({"error": "Current profile's avatar is empty"}), 400

    # Decode the current profile's avatar
    try:
        current_img = decode_base64_image(current_avatar_base64)
    except Exception as e:
        return jsonify({"error": f"Failed to decode current profile's avatar: {str(e)}"}), 400

    # Get all other profiles
    all_profiles = get_all_profiles()
    results = []

    for profile in all_profiles:
        avatar_base64 = profile.get('avatar_base64')
        if not avatar_base64 or avatar_base64.strip() == "":
            # Skip profiles with an empty or None avatar
            continue

        # Decode the other profile's avatar
        try:
            comparison_img = decode_base64_image(avatar_base64)
        except Exception as e:
            # Skip if decoding fails
            continue

        # Perform face verification
        try:
            verification_result = DeepFace.verify(current_img, comparison_img, model_name='Facenet', enforce_detection=False)
            deepface_score = 1 - verification_result['distance']
        except Exception as e:
            # Handle any errors in face verification
            continue

        # Convert images to grayscale for SSIM calculation
        gray_current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2GRAY)

        # Resize the comparison image to match the dimensions of the current image
        gray_comparison_img = cv2.resize(gray_comparison_img, (gray_current_img.shape[1], gray_current_img.shape[0]))

        # Calculate the SSIM score
        ssim_score, _ = compare_ssim(gray_current_img, gray_comparison_img, full=True)

        # Append the results
        results.append({"username": profile['username'], "deepface_score": deepface_score, "ssim_score": ssim_score})

    # Sort results based on similarity scores (higher SSIM scores indicate higher similarity)
    results = sorted(results, key=lambda x: (x['deepface_score'], -x['ssim_score']))

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
