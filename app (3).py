from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, util
import pickle
import torch

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages and session

# Database connection
def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',  # Update with your MySQL username
        password='root',  # Update with your MySQL password
        database='user_db'  # Update with your database name
    )
    return connection

# Load precomputed embeddings and sentences
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Connect to database and check for existing username or email
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            # If user or email exists, flash a message
            flash('User already exists with this username or email. Please try again.', 'danger')
            cursor.close()
            connection.close()
            return redirect(url_for('register'))

        # Correct password hashing method: pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert new user into the database
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        connection.commit()
        cursor.close()
        connection.close()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')



# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Connect to database and fetch user
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user and check_password_hash(user[3], password): 
            session['user_id'] = user[0]  # Store user ID in session
            return redirect(url_for('recommendation'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

# Recommendation route
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

    recommended_papers = []
    
    if request.method == 'POST':
        paper_title = request.form['paper_title']
        
        if paper_title.strip() == "":
            flash("Please enter a valid paper title.", 'warning')
        else:
            # Compute cosine similarity
            input_embedding = model.encode(paper_title)
            cosine_scores = util.cos_sim(embeddings, input_embedding)

            # Get the top 5 similar papers
            top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
            
            # Prepare the recommended papers
            recommended_papers = [sentences[i.item()] for i in top_similar_papers.indices.flatten()]

    return render_template('recommendation.html', recommended_papers=recommended_papers)

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user from session
    flash("You have been logged out.", 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
