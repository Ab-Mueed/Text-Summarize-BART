from flask import Flask, request, jsonify
from flask_cors import CORS
from model.summarizer import Summarizer

app = Flask(__name__)
CORS(app)
summarizer = Summarizer()

@app.route('/summarize', methods=['POST', 'GET'])
def summarize_text():
    data = request.get_json()
    text = data['text']
    summary = summarizer.summarize(text)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
