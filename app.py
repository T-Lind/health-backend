from langchain_core.messages import HumanMessage, AIMessage

from model import SuicideRiskClassifier
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
from chat import Chatbot

if not load_dotenv(".env", override=True):
    raise FileNotFoundError("No .env file found!")

ver = os.getenv("ENDPOINT_VERSION")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
jwt = JWTManager(app)

suicide_risk_classifier = SuicideRiskClassifier(max_length=1024)

chatbot = Chatbot()

logger.warning("Using random classifier, replace with full classifier!")

connection_string = os.getenv('DATABASE_URL')

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


@app.route(f'/api/{ver}/register', methods=['POST'])
@db_operation
def register(conn):
    data = request.json
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT id FROM users WHERE username = %s OR email = %s",
                (data['username'], data['email']))
    if cur.fetchone():
        return jsonify({"error": "Username or email already exists"}), 400

    hashed_password = generate_password_hash(data['password'])

    cur.execute("""
        INSERT INTO users (username, email, password_hash, role, background)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (data['username'], data['email'], hashed_password, 'user', data.get('background', '')))

    new_user_id = cur.fetchone()['id']

    access_token = create_access_token(identity=new_user_id)

    cur.close()
    return jsonify({
        "message": "User registered successfully",
        "id": new_user_id,
        "access_token": access_token
    }), 201


@app.route(f'/api/{ver}/login', methods=['POST'])
@db_operation
def login(conn):
    data = request.json
    cur = conn.cursor(cursor_factory=RealDictCursor)

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


@app.route(f'/api/{ver}/ml-predictions', methods=['POST'])
@jwt_required()
def generate_ml_prediction():
    data = request.json
    prediction = suicide_risk_classifier.predict(data['input_text'])
    return jsonify({"classification": prediction})



def get_fmtted_msgs(conn, user_id) -> list[HumanMessage | AIMessage]:
    """ Helper function for new messages sent. """
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT is_ai_response, message
        FROM chat_messages
        WHERE user_id = %s
        ORDER BY timestamp ASC
    """, (user_id,))

    chat_history = cur.fetchall()
    cur.close()

    # use langchain HumanMessage/AIMessage
    formatted_messages = []
    for msg in chat_history:
        if msg['is_ai_response']:
            formatted_messages.append(AIMessage(content=msg['message']))
        else:
            formatted_messages.append(HumanMessage(content=msg['message']))

    return formatted_messages

@app.route(f'/api/{ver}/chat', methods=['POST'])
@jwt_required()
@db_operation
def send_chat_message(conn):
    user_id = get_jwt_identity()
    data = request.json
    message = data['message']
    exchange = data.get('exchange', '')
    classification = data.get('classification', '')

    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch the user's background -- this can really help answer questions.
    cur.execute("SELECT background FROM users WHERE id = %s", (user_id,))
    user_background = cur.fetchone()['background']

    fmtted_msg = chatbot.fmt_message(message, exchange=exchange, classification=classification, background=user_background)
    messages = get_fmtted_msgs(conn, user_id)
    ai_response = chatbot.get_response(messages + [fmtted_msg])

    cur.execute("""
        INSERT INTO chat_messages (user_id, message, is_ai_response, context)
        VALUES (%s, %s, %s, %s)
        RETURNING id, timestamp
    """, (user_id, message, False, exchange))
    user_message = cur.fetchone()

    cur.execute("""
        INSERT INTO chat_messages (user_id, message, is_ai_response, context)
        VALUES (%s, %s, %s, %s)
        RETURNING id, timestamp
    """, (user_id, ai_response, True, exchange))
    ai_message = cur.fetchone()

    cur.close()

    return jsonify({
        "response": ai_response
    })


@app.route(f'/api/{ver}/chat-history', methods=['GET'])
@jwt_required()
@db_operation
def get_chat_history(conn):
    user_id = get_jwt_identity()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT id, message, is_ai_response, timestamp, context
        FROM chat_messages
        WHERE user_id = %s
        ORDER BY timestamp ASC
    """, (user_id,))

    chat_history = cur.fetchall()
    cur.close()

    return jsonify([{
        "role": "assistant" if msg['is_ai_response'] else "user",
        "content": msg['message']
    } for msg in chat_history])


@app.route(f'/api/{ver}/clear-chat', methods=['POST'])
@jwt_required()
@db_operation
def clear_chat_history(conn):
    user_id = get_jwt_identity()
    cur = conn.cursor()

    cur.execute("DELETE FROM chat_messages WHERE user_id = %s", (user_id,))
    deleted_count = cur.rowcount
    cur.close()

    return jsonify({"message": f"Deleted {deleted_count} messages from chat history"})


if __name__ == '__main__':
    app.run(debug=True)
