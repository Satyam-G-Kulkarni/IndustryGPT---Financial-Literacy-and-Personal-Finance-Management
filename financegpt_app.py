# -*- coding: utf-8 -*-
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import pipeline
from pyngrok import ngrok

def load_model_with_adapters(
    model_name,
    adapter_weights_path,
    device_map='auto',
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    use_nested_quant=True
):
    # Load appropriate compute dtype (either FP16 or BF16)
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Set up BitsAndBytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load the pre-trained model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )

    # Load the adapter weights using PeftModel
    model = PeftModel.from_pretrained(model, adapter_weights_path)

    model.config.use_cache = False  # Disable cache for inference

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Align padding
    tokenizer.padding_side = "right"

    return model, tokenizer

# Load the model and tokenizer with adapter weights
model_name = "NousResearch/Llama-2-7b-chat-hf"
adapter_weights_path = "Finance_Chatbot"
model, tokenizer = load_model_with_adapters(model_name, adapter_weights_path)

def perform_inference(model, tokenizer, input_text):
    # Create a text generation pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)

    # Format the prompt with special tokens
    prompt = f"<s>[INST] {input_text} [/INST]"

    # Generate output
    result = pipe(prompt)

    # Extract the generated text from the result
    output_text = result[0]['generated_text']

    return output_text

# Set your unique Ngrok auth token
NGROK_AUTH_TOKEN = "2gorxOm9YlGFf8MoRgQfVzx6RqY_de56fyN9P51du4UayK37"  # Replace with your actual Ngrok auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Define the HTML template for the chatbot interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .chat-container {
            width: 80%;
            max-width: 800px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .chat-box {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .user {
            background-color: #e0f7fa;
            align-self: flex-end;
        }
        .bot {
            background-color: #f1f8e9;
            align-self: flex-start;
        }
        .input-box {
            display: flex;
        }
        .input-box input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .input-box button {
            padding: 10px;
            border: none;
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 10px;
        }
        .input-box button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-box" id="chat-box"></div>
        <div class="input-box">
            <input type="text" id="user-input" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        // Function to escape potentially harmful characters in the bot's response
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const message = input.value.trim();
            if (message === '') return;

            // Add user message to chat box
            chatBox.innerHTML += '<div class="message user">' + escapeHtml(message) + '</div>';
            input.value = '';

            // Send message to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: message })
            });

            const data = await response.json();

            // Add bot response to chat box (sanitize any HTML)
            chatBox.innerHTML += '<div class="message bot">' + escapeHtml(data.response) + '</div>';
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return html_template

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's prompt from the request
    user_input = request.json.get('prompt')

    # Generate response using the perform_inference function
    bot_response = perform_inference(model, tokenizer, user_input)

    # Return the response as JSON
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(" * Running on", public_url)

    # Run Flask app
    app.run()