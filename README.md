# Capstone Project: Cloned Profile Detection on Social Networks

Welcome to the repository for our capstone project. This project focuses on detecting cloned profiles on a social network using network graphs. 

## Objective
The main objective of this project is to identify cloned profiles on social networks using machine learning algorithms. By simulating the behavior of the network with network graphs and utilizing the Neo4j graph database, we aim to analyze and gain insights into the behavior of these cloned profiles.

## Project Components
- **Machine Learning Algorithms:** These are used to detect cloned profiles within the network.
- **Network Analysis:** Centrality measures are applied to the cloned profiles to understand their behavior and impact within the network.

## Technology Stack
- **Frontend:** React.js
- **Backend:** Flask
- **Database:** Neo4j (for graph-related functionality)

## Installation and Setup
1. **Clone the Repository**
   ```
   git clone https://github.com/Sadhika1012/SVM_511_565_577_615.git
   ```
2. **Frontend Setup (React.js)**
   - Navigate to the `frontend` directory.
   - Install dependencies:
     ```
     npm install
     ```
   - Start the React application:
     ```
     npm start
     ```

3. **Backend Setup (Flask)**
   - Navigate to the `backend` directory.
   - Create a virtual environment:
     ```
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     ```
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Start the Flask application:
     ```
     flask run
     ```

4. **Database Setup (Neo4j)**
   - Install Neo4j from the official website: [Neo4j Download](https://neo4j.com/download/)
   - Start the Neo4j database and configure it according to your project requirements.

## Usage
1. Start the backend Flask server.
2. Start the frontend React application.
3. Ensure the Neo4j database is running and properly configured.
4. Access the application through your web browser.

## Features
- **Profile Cloning Detection:** Identifies cloned profiles using machine learning algorithms.
- **Network Analysis:** Analyzes the network using various centrality measures to understand the behavior of cloned profiles.


