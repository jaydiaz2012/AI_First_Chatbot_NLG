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
System_Prompt_Forecast = """
Role:
You are SalesX AI, an AI-based Revenue Forecasting Model designed to generate predictions of future sales based on historical data. Your primary function is to produce accurate, data-driven forecasts to aid users in strategic planning.

Instructions:

Accept a list of historical sales data as input, consisting of numerical values representing revenue for past periods.
Analyze the provided historical data to identify trends, seasonality, and patterns.
Generate a sales forecast for the next 12 periods using appropriate statistical or machine learning models.
Output the forecasted values as a comma-separated string for easy parsing and into a line chart with the months as x and sales as y. 
Ensure your forecast takes into account both short-term trends and long-term patterns to improve accuracy.
Maintain clarity and conciseness in your output, focusing only on the forecasted values without extraneous information.

Context:
The user will input a series of numerical values representing revenue over a sequence of past periods (e.g., monthly sales data for the past two years). Your task is to predict the sales for the next 12 periods based on this historical data. The user will leverage your forecast for financial planning, budgeting, or inventory management.

Constraints:

Do not assume any additional data beyond what the user provides (e.g., macroeconomic factors or market conditions).
The forecasted output should be limited to 12 values, representing the next 12 periods.

Examples:

Input: [1200, 1350, 1500, 1450, 1600, 1700, 1550, 1650, 1800, 1750, 1900, 1850]
Output: 1900, 1950, 2000, 2100, 2050, 2150, 2200, 2250, 2300, 2400, 2350, 2450

Input: [100, 200, 300, 250, 350, 400, 450, 500, 550, 600, 650, 700]
Output: 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300
"""
System_Prompt_Explanation = """
You are SalesX AI, AI-based Revenue Forecast Explanation Model designed to provide clear, insightful interpretations of the forecasted values generated by the forecasting model. Your primary function is to explain the forecast results in a way that helps users understand and act upon the information.

Instructions:

Analyze the forecasted revenue values and identify significant trends, such as growth patterns, seasonality, or unexpected fluctuations.
Interpret what the forecasted values imply about future business performance, focusing on areas like sales growth, potential slowdowns, or cyclical changes.
Highlight any peaks, troughs, or irregularities that might require the user's attention.
Offer actionable insights or recommendations based on the forecasted data (e.g., adjusting inventory levels, planning marketing campaigns, or reallocating resources).
Ensure explanations are clear, concise, and tailored to the user’s needs, focusing on helping them make strategic decisions.

Context:
The forecasted revenue data you receive will be based on historical sales trends provided by the user. The user is typically interested in understanding the forecasted outcomes to make informed business decisions, optimize resource allocation, and plan for the future. Your explanations will guide the user in interpreting the forecast’s implications.

Constraints:

Do not re-run or modify the forecast calculations—focus solely on interpreting the given data.
Avoid technical jargon; your explanations should be understandable to users with limited expertise in data analysis.
Ensure that your insights are actionable and relevant to business strategy rather than purely descriptive.

Examples:

Forecasted Values: 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750
Explanation:

The forecast shows a steady upward trend, suggesting consistent growth in revenue. This could indicate increased demand or successful sales strategies.
Consider increasing inventory or expanding marketing efforts to capitalize on this growth trend.
Forecasted Values: 800, 750, 700, 680, 670, 660, 650, 640, 630, 620, 610, 600
Explanation:

A declining trend is evident, which could signal a drop in market demand or increased competition. This suggests a need to review sales strategies or explore new revenue streams.
Immediate action may be required to prevent further declines, such as introducing promotional offers or improving product differentiation.
"""

# Sidebar for API key and options
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and api_key.startswith('sk-'):  # Removed length check
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your sales data!', icon='👉')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='⚠️')

    st.header("Instructions")
    st.write("1. Enter a valid OpenAI API Key.")
    st.write("2. Click SalesX AI on the Sidebar to get started!")
    st.write("3. Input your sales data.")
    st.write("4. Click 'Forecast Sales' to see the predictions.")
    
    if st.button("Reset"):
        st.session_state.clear()  # Clear session state to reset the app

    options = option_menu(
        "Content",
        ["Home", "About Me", "SalesX AI"],
        default_index=0
    )

# Initialize session state for messages
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
                {"role": "system", "content": "You are an AI assistant analyzing user behavior data. Provide accurate statistics and insights based on the full dataset."},
                {"role": "user", "content": full_prompt}
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
            {"role": "system", "content": System_Prompt_Forecast},
            {"role": "user", "content": prompt}
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

# Function to generate explanation using OpenAI API
def generate_explanation(data, forecast):
    # Prepare the historical data for the prompt
    historical_data_str = data.to_string(index=False)  # Convert DataFrame to string for better readability
    forecast_str = ', '.join(map(str, forecast))  # Convert forecasted values to a string

    # Load and prepare data for RAG
    dataframed = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI_First_Chatbot_Project/refs/heads/main/Restaurant_revenue_final.csv')
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()

    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    # Generate embedding for the forecast string
    query_embedding = get_embedding(forecast_str, engine='text-embedding-3-small')
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Search for relevant documents
    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)

    # Modify the prompt to focus on how the forecast was derived and analyze historical trends
    prompt = f"""
    {System_Prompt_Explanation}
    
    1. Analyze the historical revenue data provided below and identify key trends, fluctuations, and patterns:
    {historical_data_str}
    
    2. Based on the historical data, explain how the forecasted revenue values were derived: {forecast_str}.
    
    3. Use the following context to enhance your analysis and explanation, but do not assume it is directly related to the user's input data:
    {context}
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
    Welcome to SalesX AI, an innovative application designed to empower businesses with data-driven revenue forecasting. In today's fast-paced market, understanding future sales trends is crucial for strategic planning, inventory management, and financial forecasting. SalesX AI leverages advanced artificial intelligence to provide accurate predictions based on historical sales data, helping businesses make informed decisions.
    """)

    st.subheader("🧿 Features")
    st.write("""
    - **User-Friendly Interface**: Navigate effortlessly through the application with a clean and intuitive design.
    - **Revenue Forecasting**: Input your historical sales data, and let SalesX AI generate forecasts for the next 12 periods, giving you a clear view of potential future revenue.
    - **Insightful Explanations**: Not only does SalesX AI provide forecasts, but it also explains how these predictions were derived, analyzing historical trends and patterns to enhance your understanding.
    - **Contextual Analysis**: The application utilizes additional contextual data to enrich the forecasting process, ensuring that your predictions are informed by relevant market insights.
    - **Automated Visualizations**: Visualize your sales data and forecasts with engaging charts, making it easier to grasp trends and make strategic decisions.
    """)

    st.subheader("🔑 Why SalesX AI?")
    st.write("""
    In an era where data is king, SalesX AI was created to bridge the gap between complex data analysis and actionable business insights. Whether you're a small business owner, a sales manager, or a financial analyst, having access to reliable forecasts can significantly impact your planning and strategy.

    SalesX  AI aims to simplify the forecasting process, allowing users to focus on what matters most—growing their business. By transforming historical sales data into clear, actionable insights, SalesX AI empowers users to anticipate market changes and adapt their strategies accordingly.
    """)

# About Us Page
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
    data_input_method = st.selectbox("How would you like to input your sales data?", ["Upload CSV", "Enter Data Manually"])

    if data_input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your sales data CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:", data.head())
            # Create a dropdown for selecting the column to forecast
            sales_column = st.selectbox("Select the column to forecast:", data.columns)
    else:
        # Manual data entry
        st.write("Enter your sales data below:")
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
