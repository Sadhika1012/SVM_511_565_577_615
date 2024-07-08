import csv
import bcrypt
import random
import string
from neo4j import GraphDatabase

# Function to generate a random password
def generate_random_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for i in range(length))

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Connect to Neo4j
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"
driver = GraphDatabase.driver(uri, auth=("neo4j", password))

def create_user_profile(tx, name, city, position, hashed_password):
    query = (
        "CREATE (u:User {name: $name, city: $city, position: $position, password: $hashed_password})"
    )
    tx.run(query, name=name, city=city, position=position, hashed_password=hashed_password)

# Read the CSV file and create profiles in Neo4j
csv_file_path = 'another_test.csv'
with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    with driver.session() as session:
        for row in reader:
            name = row['name']
            city = row['city']
            position = row['position']
            password = generate_random_password()
            hashed_password = hash_password(password)
            session.write_transaction(create_user_profile, name, city, position, hashed_password)
            print(f"Created profile for {name} with password: {password}")

# Close the driver connection
driver.close()
