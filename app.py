import os
import openai
import numpy as np
import pandas as pd
import json
import io 
from io import StringIO
import matplotlib.pyplot as plt 
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import warnings
from PIL import Image
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import base64
import random
from datetime import datetime, timedelta
import requests
import json

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 SalesX AI", layout="wide")
System_Prompt = """
Role:
You are a SalesX Ai, a highly skilled AI assistant specializing in sales data analysis and time-series forecasting. Your expertise lies in analyzing historical sales data, identifying trends, and predicting sales for the next 12 months with high accuracy.

Instructions:
Your purpose is to assist businesses by providing actionable insights through accurate sales forecasts, identifying seasonality, trends, and anomalies, and generating visual reports to support decision-making.
Generate a sales forecast for the next 12 periods using appropriate statistical or machine learning models.
Output the forecasted values as a comma-separated string for easy parsing and into a line chart with the months as x and sales as y. 

Context:
The users will provide historical sales data. The assistant will preprocess this data, train forecasting models, and present outputs in user-friendly formats such as charts, graphs, or downloadable reports. Users may ask questions in natural language, request forecasts for specific products or regions, or explore hypothetical scenarios.

Constraints:

Ensure forecasts are based on rigorous analysis, including cleaning and preprocessing data for accuracy.
Use state-of-the-art algorithms such as ARIMA, LSTM, or Prophet, depending on the dataset’s characteristics.
Provide results within a short response time, ensuring clarity and interpretability for non-technical users.
Include confidence intervals to account for uncertainty in forecasts.
Visualizations must be clear and tailored to user queries.
Do not assume any additional data beyond what the user provides (e.g., macroeconomic factors or market conditions).
The forecasted output should be limited to 12 values, representing the next 12 periods.

End Goal:
Deliver actionable, data-driven sales forecasts and insights that empower businesses to make informed decisions about inventory, budgeting, and strategic planning. Ensure the system is user-friendly, scalable, and adaptable to diverse business needs.

Examples:

Input: [12270, 5860, 10390, 18418, 10191, 16964, 16284, 10734, 11265, 5466, 18526, 9426]
Output: 19000, 19050, 20000, 21000, 20050, 21050, 20200, 20250, 20300, 24000, 23050, 20450

Input: [12270, 5860, 10390, 18418, 10191, 16964, 16284, 10734, 11265, 5466, 18526, 9426]
Output: 11250, 12800, 84550, 97600, 23950, 12000, 231050, 13100, 21150, 12400, 31250, 21300
"""

# Sidebar for API key and options
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and api_key.startswith('sk-'): 
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your sales data!', icon='✔️')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='❌')

    st.header("Warning!🛑")
    st.subheader("Do the following to enable SalesX to work successfully.")
    st.write("Enter a valid OpenAI key.")
    st.write("Use only CSV files when uploading. You can enter manually the data.")
    st.write("If using CSV, select the column with the header Sales or Revenue or the like for analysis.")
    st.write("After generating, the AI Bot will provide a table of forecasted sales for the next 12 months starting at 0, a line chart, and a summary.")

    options = option_menu(
        "Content",
        ["Home", "About Me", "SalesX AI"],
        default_index=0
    )


if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_nlg_response(prompt, forecast):
    """
    Generate text using OpenAI's GPT model for NLG.
    """
    try:
        # Prepare a summary of the data
        data_summary = forecast        
        full_prompt = f"""Analyze the following dataset:

Summary Statistics:
{data_summary}

Now, based on this data, {prompt}

Provide a detailed analysis, including exact counts and percentages where applicable."""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Using a model with larger context
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing sales data. Provide accurate statistics and insights based on the full dataset."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error in generating NLG response: {str(e)}")
        return "Sorry, I couldn't generate a response at this time."

# Function to forecast revenue
def forecast_sales(data, sales_column):
    # Prepare the input for the GPT model
    sales_data = data[sales_column].tolist()
    sales_data_str = ', '.join(map(str, sales_data))

    # Create a prompt for the GPT model
    prompt = f"Given the following sales data: {sales_data_str}, forecast the next 12 periods of revenue. Return only the forecasted values as a comma-separated string."

    # Call the OpenAI API to generate the forecast
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.1,
        messages=[
            {"role": "system", "content": System_Prompt},
            {"role": "user", "content": full_prompt}
        ]
    )

    # Extract the forecasted values from the response
    forecasted_values = response['choices'][0]['message']['content']

    # Print the response for debugging
    print("API Response:", forecasted_values)

    # Convert the forecasted values to a list of floats
    try:
        forecasted_data = [float(value) for value in forecasted_values.split(',')]
    except ValueError as e:
        st.error("Error parsing forecasted values. Please check the API response.")
        print("Error:", e)
        return None

    return forecasted_data

def generate_explanation(data, forecast):
    historical_data_str = data.to_string(index=False)
    forecast_str = ', '.join(map(str, forecast)) 

    dataframed = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI_First_Chatbot_Project/refs/heads/main/Restaurant_revenue_final.csv')
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()

    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    query_embedding = get_embedding(forecast_str, engine='text-embedding-3-small')
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)

    prompt = f"""
    {System_Prompt}
    Based on the given sales data and forecast results, craft a concise and informative response that communicates the insights effectively: {historical_data_str}. 
    Ensure the response is tailored to the user's query, uses a professional tone, and includes specific details such as time periods, trends, and actionable recommendations: {forecast_str}. 
    Provide context for the predictions, explain any significant anomalies or changes, and use simple language to make the insights accessible to non-technical users. 
    If applicable, suggest strategies for improving sales performance: {context}.

    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.7,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are an AI assistant analyzing sales data. Provide accurate statistics and insights based on the full dataset."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content'].stip()

# Home Page
if options == "Home":
    st.title("Welcome to SalesX AI!🏆")
    st.write("""
    Introducing SalesX AI, a cutting-edge application designed to help businesses harness the power of data-driven revenue forecasting. In today’s rapidly evolving market, gaining insights into future sales trends is essential for effective strategic planning, inventory management, and financial forecasting. By utilizing advanced artificial intelligence, SalesX AI delivers precise predictions based on historical sales data, enabling businesses to make informed, data-backed decisions.
    """)
    
    st.subheader("Steps To Forecasting")
    st.write("""
    - Data Cleaning: Remove duplicate and null records. Handle missing values using imputation techniques or exclude incomplete rows if necessary.
    - Feature Extraction: Identify key features like product category, sales region, price points, and promotional periods.
    - Data Transformation: Normalize or standardize data for better model performance. Aggregate data into time intervals (e.g., daily, weekly, monthly).
    - Outlier Detection: Use statistical techniques or models like Isolation Forest to spot anomalies.    
    """)

# About Me Page
elif options == "About Me":
    st.title("About Me")
    My_image = Image.open("images/photo-me1.jpg")
    my_resized_image = My_image.resize((180,180))
    st.image(my_resized_image)
    st.write("I am Jeremie Diaz, an AI builder and programmer.")
    st.write("Don't hesitate to contact me on my LinkedIn and check out my other projects on GitHub!")
    st.write("https://www.linkedin.com/in/jandiaz/")
    st.write("https://github.com/jaydiaz2012/")


# Forecast Page
elif options == "SalesX AI":
    st.title("📈 SalesX AI")
    
    # Option for user to input data
    data_input_method = st.selectbox("Upload Sales Data Here (CSV only) or Manually Input Sales Data", ["CSV", "Manual Data"])

    if data_input_method == "CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Sales Data Preview:", data.head())
            sales_column = st.selectbox("Select the (Sales or Revenue) column to forecast:", data.columns)
    else:
        st.write("Manually input sales data below:")
        sales_data = st.text_area("Sales Data (comma-separated, e.g., 100, 150, 200)", "")
        if sales_data:
            sales_list = [float(x) for x in sales_data.split(",")]
            data = pd.DataFrame({'Sales': sales_list})
            sales_column = 'Sales'  # Set default sales column for manual entry

    if 'data' in locals() and 'sales_column' in locals():
        if st.button("Forecast Sales"):
            forecast = forecast_sales(data, sales_column)
            st.write("Forecasted Sales:", forecast)

            #explanation = generate_explanation(data, forecast)
            #st.write("Explanation:", explanation)

            # Visualization
            st.header("Forecast Sales Chart")
            st.line_chart(forecast)
    
            #NLG
            prompt = f"Analyze the {forecast}. Provide insights on the trend."
            nlg_response = generate_nlg_response(prompt, forecast)
            st.write("Forecast Sales:", nlg_response)

    #NLP Page
    def initialize_conversation(prompt):
        if 'message' not in st.session_state:
            st.session_state.message = []
            st.session_state.message.append({"role": "system", "content": System_Prompt})

    initialize_conversation(System_Prompt)

    for messages in st.session_state.message:
        if messages['role'] == 'system':
            continue
        else:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

    if user_message := st.chat_input("Ask me more about your forecast!"):
        with st.chat_message("user"):
            st.markdown(user_message)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=st.session_state.message,
        )
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})
