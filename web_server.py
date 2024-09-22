from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory

# Set up the app and file upload settings
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf'}

# Create the uploads directory if not exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Route to serve styles.css
@app.route('/static/assets/css/styles.css')
def serve_styles():
    return send_from_directory('static/assets/css', 'styles.css')

# Route to serve app.js
@app.route('/static/assets/js/app.js')
def serve_js():
    return send_from_directory('static/assets/js', 'app.js')


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


# Upload documents
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return 'No selected file.'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path})
    else:
        return 'File type not allowed.'


# Extract text from the document
@app.route('/extract', methods=['POST'])
def extract_text():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    # Here you can process the file to extract text (using your extraction function)
    # For demonstration, just read raw text for now if it’s a .txt file
    extracted_text = ''
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            extracted_text = f.read()
    elif file_path.endswith('.pdf'):
        # Add your PDF extraction logic here
        extracted_text = "PDF extraction not yet implemented"

    # You can store this text in a database or use it directly for LLM training
    return jsonify({"extracted_text": extracted_text})


# Endpoint to trigger training on extracted text
@app.route('/train', methods=['POST'])
def train_model():
    text_data = request.json.get('text_data')

    if not text_data:
        return jsonify({"error": "No text data provided"}), 400

    # Insert your logic to train your LLM model on the extracted text here
    # For now, we'll mock it
    # Train the model with `text_data`

    return jsonify({"message": "Training initiated", "status": "success"})


if __name__ == '__main__':
    app.run(debug=True, port=9901, host="localhost")
