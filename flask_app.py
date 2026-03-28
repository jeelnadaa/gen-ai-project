import os
import uuid
import json
import threading
import time
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from app import process_pdf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './.tmp_flask_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# In-memory store for task status and results
# In a production app, use Redis/Postgres
tasks = {}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get settings from form
    groq_model = request.form.get('groq_model', 'llama-3.3-70b-versatile')
    max_clauses = request.form.get('max_clauses', type=int)
    min_clause_length = request.form.get('min_clause_length', type=int, default=30)
    reference_summary = request.form.get('reference_summary', '')

    task_id = str(uuid.uuid4())
    filename = secure_filename(f"{task_id}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    tasks[task_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Initializing pipeline...",
        "result": None,
        "error": None
    }

    def background_processing(tid, fpath, model, m_clauses, min_len, ref_sum):
        try:
            def progress_callback(processed, total):
                progress = int((processed / total) * 100)
                tasks[tid]["progress"] = progress
                tasks[tid]["status"] = "processing"
                tasks[tid]["message"] = f"Processing clause {processed}/{total}..."

            result = process_pdf(
                fpath,
                groq_model=model,
                max_clauses=m_clauses if m_clauses and m_clauses > 0 else None,
                reference_summary=ref_sum.strip() or None,
                min_clause_length=min_len,
                progress_callback=progress_callback
            )
            tasks[tid]["status"] = "completed"
            tasks[tid]["progress"] = 100
            tasks[tid]["message"] = "Processing complete!"
            tasks[tid]["result"] = result
        except Exception as e:
            tasks[tid]["status"] = "error"
            tasks[tid]["error"] = str(e)
            tasks[tid]["message"] = f"Error: {str(e)}"

    thread = threading.Thread(
        target=background_processing,
        args=(task_id, filepath, groq_model, max_clauses, min_clause_length, reference_summary)
    )
    thread.start()

    return jsonify({"task_id": task_id})

@app.route('/status/<task_id>')
def status(task_id):
    def generate():
        last_progress = -1
        last_status = ""
        
        while True:
            if task_id not in tasks:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                break
            
            task = tasks[task_id]
            # Only send update if status or progress changed
            if task['progress'] != last_progress or task['status'] != last_status:
                data = {
                    "status": task['status'],
                    "progress": task['progress'],
                    "message": task['message'],
                    "error": task['error'],
                    "result": task['result'] if task['status'] == 'completed' else None
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = task['progress']
                last_status = task['status']

            if task['status'] in ['completed', 'error']:
                break
            
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<task_id>')
def download(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return "Result not ready", 404
    
    result = tasks[task_id]['result']
    return Response(
        json.dumps(result, indent=2, ensure_ascii=False),
        mimetype='application/json',
        headers={"Content-disposition": f"attachment; filename=results_{task_id}.json"}
    )

if __name__ == '__main__':
    # Ensure templates and static exist
    app.run(debug=True, port=5000)
