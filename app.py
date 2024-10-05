from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import json
from datetime import datetime
from model import SuicideRiskClassifier, RandomClassifier
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import logging
from functools import wraps

load_dotenv(".env", override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

# suicide_risk_classifier = SuicideRiskClassifier(max_length=1024)
suicide_risk_classifier = RandomClassifier()

logger.warning("Using random classifier, replace with full classifier!")


# Get the connection string from the environment variable
connection_string = os.getenv('DATABASE_URL')

# Create a connection pool
connection_pool = ThreadedConnectionPool(5, 20, connection_string)

def get_db_connection():
    return connection_pool.getconn()

def return_db_connection(conn):
    connection_pool.putconn(conn)

def db_operation(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        conn = get_db_connection()
        try:
            result = f(conn, *args, **kwargs)
            conn.commit()
            return result
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            return jsonify({"error": "An error occurred while processing your request"}), 500
        finally:
            return_db_connection(conn)
    return decorated_function

@app.route('/api/register', methods=['POST'])
@db_operation
def register(conn):
    data = request.json
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Check if username or email already exists
    cur.execute("SELECT id FROM users WHERE username = %s OR email = %s",
                (data['username'], data['email']))
    if cur.fetchone():
        return jsonify({"error": "Username or email already exists"}), 400

    # Hash the password
    hashed_password = generate_password_hash(data['password'])

    # Insert new user
    cur.execute("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (data['username'], data['email'], hashed_password, 'user'))

    new_user_id = cur.fetchone()['id']

    # Create access token
    access_token = create_access_token(identity=new_user_id)

    cur.close()
    return jsonify({
        "message": "User registered successfully",
        "id": new_user_id,
        "access_token": access_token
    }), 201

@app.route('/api/login', methods=['POST'])
@db_operation
def login(conn):
    data = request.json
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Find user by username or email
    cur.execute("SELECT * FROM users WHERE username = %s OR email = %s",
                (data.get('username', ''), data.get('email', '')))
    user = cur.fetchone()

    if user and check_password_hash(user['password_hash'], data['password']):
        access_token = create_access_token(identity=user['id'])
        cur.close()
        return jsonify({
            "message": "Logged in successfully",
            "access_token": access_token,
            "user_id": user['id'],
            "username": user['username'],
            "role": user['role']
        })
    else:
        cur.close()
        return jsonify({"error": "Invalid username/email or password"}), 401

@app.route('/api/v1/ml-predictions', methods=['POST'])
def generate_ml_prediction():
    data = request.json
    prediction = suicide_risk_classifier.predict(data['input_text'])
    return jsonify({"classification": prediction})


if __name__ == '__main__':
    app.run(debug=True)