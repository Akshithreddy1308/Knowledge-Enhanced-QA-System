from flask import Flask, request, render_template, jsonify
from retrieval_pipeline import get_answer, dialogue_history

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query')
    if query_text:
        answer = get_answer(query_text)
        return jsonify({'response': answer, 'history': dialogue_history})
    return jsonify({'response': 'No query provided'})

if __name__ == '__main__':
    app.run(debug=True)