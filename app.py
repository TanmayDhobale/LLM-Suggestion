from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

def classify_query(query):

    return "gpt2"

def suggest_llm(query_type):

    return "gpt2"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_query = request.form['text']
    query_type = classify_query(user_query)
    suggested_llm = suggest_llm(query_type)
    
    model = AutoModelForCausalLM.from_pretrained(suggested_llm)
    tokenizer = AutoTokenizer.from_pretrained(suggested_llm)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

    response = generator(user_query, max_length=100, truncation=True)
    return jsonify(response[0]['generated_text'])

if __name__ == '__main__':
    app.run(debug=True)
