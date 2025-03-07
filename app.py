import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import sqlite3
import pandas as pd
import pickle

# Load a lightweight open-source model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Smaller model for better RAM efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Reduce memory usage
    low_cpu_mem_usage=True  # Optimize loading
).to(device)

# Load additional machine learning model (if exists)
try:
    with open("model.pkl", "rb") as f:
        ml_model = pickle.load(f)
except FileNotFoundError:
    ml_model = None

# Function to generate responses using the model
def generate_response(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to query the database
def query_database(query):
    conn = sqlite3.connect('case_management.db')
    cursor = conn.cursor()
    
    try:
        if "most common case types" in query.lower():
            cursor.execute("SELECT case_type, COUNT(*) as count FROM cases GROUP BY case_type ORDER BY count DESC")
            results = cursor.fetchall()
            response = "Most common case types:\n" + "\n".join([f"- {r[0]}: {r[1]} cases" for r in results])
        elif "assignee resolution rates" in query.lower():
            cursor.execute("SELECT assignee, AVG(outcome) as resolution_rate FROM cases GROUP BY assignee ORDER BY resolution_rate DESC")
            results = cursor.fetchall()
            response = "Assignee resolution rates:\n" + "\n".join([f"- {r[0]}: {r[1]:.2f} resolution rate" for r in results])
        else:
            response = generate_response(query)
    except Exception as e:
        response = f"Database error: {str(e)}"
    
    conn.close()
    return response

# Function to handle Gradio interface input
def chatbot_interface(query):
    if any(keyword in query.lower() for keyword in ["case types", "resolution rates", "factors affecting"]):
        return query_database(query)
    return generate_response(query)

# Launch Gradio interface
def launch_gradio():
    interface = gr.Interface(
        fn=chatbot_interface,
        inputs="text",
        outputs="text",
        title="Case Management Chatbot",
        description="Ask questions about case trends, resolution times, and more."
    )
    interface.launch(share=True)  # `share=True` to allow external access on Colab

# Run the Gradio app
if __name__ == "__main__":
    launch_gradio()
