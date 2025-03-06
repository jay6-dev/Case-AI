import sqlite3
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

# Load the trained model from the .pkl file
def load_trained_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Connect to the SQLite database
def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

# Function to execute SQL queries
def execute_sql_query(query, conn):
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    return results

# Load the case management data
def load_case_data(db_path):
    conn = connect_db(db_path)
    cases_df = pd.read_sql_query("SELECT * FROM cases", conn)
    clients_df = pd.read_sql_query("SELECT * FROM clients", conn)
    merged_df = pd.merge(cases_df, clients_df, on="client_id", how="inner")
    conn.close()
    return merged_df

# Function to predict case outcome using the trained model
def predict_case_outcome(model, case_details):
    # Convert case_details into a DataFrame
    input_data = pd.DataFrame([case_details])
    prediction = model.predict(input_data)
    return prediction[0]

# Initialize the smaller model (GPT-2) for natural language queries
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_nlp = AutoModelForCausalLM.from_pretrained("gpt2")

# Chatbot function
def chatbot(query, case_data, trained_model):
    # Handle natural language queries
    if "predict" in query.lower():
        # Extract case details from the query (this is a placeholder; you'll need to parse the query)
        case_details = {
            "case_type": "Billing",  # Example field
            "priority": "High",       # Example field
            "client_age": 30,         # Example field
        }
        outcome = predict_case_outcome(trained_model, case_details)
        return f"The predicted outcome for this case is: {outcome}"
    
    # Handle general questions about case trends
    elif "common case types" in query.lower():
        common_case_types = case_data["case_type"].value_counts().to_string()
        return f"The most common case types are:\n{common_case_types}"
    
    elif "resolution time factors" in query.lower():
        factors = case_data.corr()["resolution_time"].sort_values(ascending=False).to_string()
        return f"The factors affecting resolution time are:\n{factors}"
    
    elif "assignee resolution rates" in query.lower():
        assignee_rates = case_data.groupby("assignee")["outcome"].mean().sort_values(ascending=False).to_string()
        return f"The assignee resolution rates are:\n{assignee_rates}"
    
    else:
        # Use GPT-2 for general conversational responses
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model_nlp.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Gradio interface
def chatbot_interface(query):
    # Load the case management data
    case_data = load_case_data("case_management.db")
    
    # Load the trained model
    trained_model = load_trained_model("model.pkl")
    
    # Get the chatbot response
    response = chatbot(query, case_data, trained_model)
    return response

# Launch the Gradio app
gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Case Management Chatbot",
    description="Ask questions about case trends, resolution times, and more."
).launch()