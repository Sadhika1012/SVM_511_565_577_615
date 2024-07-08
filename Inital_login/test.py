from flask import Flask, request, render_template, redirect, url_for, session as flask_session, flash
from neo4j import GraphDatabase
import bcrypt
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a secure, random secret key

# Connect to Neo4j
uri = "neo4j+s://77efce7f.databases.neo4j.io:7687"
username = "neo4j"
password = "OIcHAuEMLZfaHcauVn4wK2Itx6db8iz9IaL52hHFsDs"
driver = GraphDatabase.driver(uri, auth=("neo4j", password))

def find_user(tx, name):
    query = "MATCH (u:User {name: $name}) RETURN u"
    result = tx.run(query, name=name)
    return result.single()

@app.route('/')
def index():
    if 'user' in flask_session:
        return redirect(url_for('profile'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password'].encode('utf-8')
        
        with driver.session() as neo4j_session:
            user = neo4j_session.execute_read(find_user, name)
            if user:
                user_data = user['u']
                if bcrypt.checkpw(password, user_data['password'].encode('utf-8')):
                    flask_session['user'] = {
                        'name': user_data['name'],
                        'city': user_data['city'],
                        'position': user_data['position']
                    }
                    return redirect(url_for('profile'))
            flash('Invalid name or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'user' not in flask_session:
        return redirect(url_for('login'))
    user = flask_session['user']
    return render_template('profile.html', user=user)

@app.route('/logout')
def logout():
    flask_session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
