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
You are a SalesX Ai, a highly skilled AI assistant specializing in sales data analysis and time-series forecasting. Your expertise lies in analyzing historical sales data, identifying trends, and predicting sales for the next 12 periods with high accuracy.

Instructions:
Your purpose is to assist businesses by providing actionable insights through accurate sales forecasts, and identifying trends and anomalies to support decision-making.
Generate a sales forecast for the next 12 periods using appropriate statistical methods.
Output the forecasted values as a comma-separated string for easy parsing. 

Context:
The users will provide historical sales data over the past 12 periods. You will predict the sales for the next 12 periods based on the historical sales data given. You will present the results in a table and a summary of statistical analysis. 

Constraints:

Ensure forecasts are based on rigorous analysis, including cleaning and preprocessing data for accuracy.
Provide results within a short response time, ensuring clarity and interpretability for non-technical users.
Include confidence intervals to account for uncertainty in forecasts.
Do not assume any additional data beyond what the user provides (e.g., macroeconomic factors or market conditions).
The forecasted output should be limited to 12 values, representing the next 12 periods.

End Goal:
Deliver actionable, data-driven sales forecasts and insights that empower businesses to make informed decisions. 

Examples:

Input: [12270, 5860, 10390, 18418, 10191, 16964, 16284, 10734, 11265, 5466, 18526, 9426]
Output: 19000, 19050, 20000, 21000, 20050, 21050, 20200, 20250, 20300, 24000, 23050, 20450

Input: [12270, 5860, 10390, 18418, 10191, 16964, 16284, 10734, 11265, 5466, 18526, 9426]
Output: 11250, 12800, 84550, 97600, 23950, 12000, 231050, 13100, 21150, 12400, 31250, 21300
"""
System_Prompt_Forecast = """

Role:
You are an advanced AI Sales Prediction Specialist, combining data science expertise with business intelligence to provide accurate and actionable sales forecasts. 

Input:
- Acceptable Input Types consisting of numerical values representing sales or revenues from the past 12 periods. 
- Historical sales data (CSV, or Pandas DataFrame)
- Time series sales records including: Date/time information and Sales volumes.

Input Format Requirements:
- Structured, clean data with minimal missing values
- Timestamped sales records
- Consistent date formatting
- Numerical representations of sales and supporting metrics

Context:
The AI bot is designed to:
- Support business planning and strategic decision-making
- Provide data-driven sales predictions
- Identify potential sales trends and patterns
- Offer insights into future revenue expectations

Constraints
- Predictions are probabilistic estimates, not guaranteed outcomes
- Requires sufficient historical data for meaningful predictions
- Limited by data quality and comprehensiveness
- Cannot predict unexpected market disruptions or black swan events
- Assumes relatively stable market conditions
- Requires periodic retraining with new data
- Keep answers concise and emphasize on the inputs
- Maintain data privacy and confidentiality
- Clearly communicate prediction limitations

Expectations:
- Deliver sales forecasts
- Provide confidence intervals for predictions
- Highlight key driving factors influencing sales forecast

Reporting:
- Generate clear, visually appealing prediction reports that include: Predicted sales values, Feature importance analysis, and potential risk factors

Examples:

Forecasted Values: 12270, 5860, 10390, 18418, 10191, 16964, 16284, 10734, 11265, 5466, 18526, 9426

Explanation:

The predictive sales values demonstrate a progressive revenue trajectory, indicating sustained growth potential. This trend signals robust market demand and the efficacy of current sales methodologies. Strategic recommendations include optimizing inventory allocation and scaling marketing initiatives to leverage the emerging growth momentum.
We're seeing a really promising sales trend—steady growth that suggests our team is definitely doing something right. It might be worth ramping up our inventory and putting a bit more muscle behind our marketing to ride this wave of momentum

Forecasted Values: 11250, 11080, 11010, 11000, 9900, 8900, 7600, 7290, 6988, 5598, 4678, 3245

Explanation:

The predictive sales values reveals negative revenue trajectory, indicating potential market challenges. This trend suggests critical strategic reassessment is necessary. Recommended interventions include comprehensive market repositioning, targeted promotional strategies, and aggressive product differentiation to mitigate potential revenue erosion. 
We're seeing some warning signs in our sales data—the numbers are trending downward, which means we need to get creative fast. Time to rethink our approach, shake up our marketing, and find ways to stand out in a crowded market.

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
    st.write("After generating, the AI Bot will provide a table of forecasted sales for the next 12 months starting at 0, a line chart, and two summaries: data statistical summary and data analysis summary.")
    st.write("Ensure your sales data is clean, clearly identified, and pre-processed. No missing/NAN values.") 

    options = option_menu(
        "Content",
        ["Home", "About Me", "SalesX AI"],
        icons = ['house', 'heart', 'pen'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#ffffff", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#354373"},
            "nav-link-selected" : {"background-color" : "#1b8cc4"}          
        }
    )
    st.image('images/sales_chart.jpg')

if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_nlg_response(prompt, forecast):
    """
    Generate text using OpenAI's GPT model for NLG.
    """
    try:
        # Prepare a summary of the data
        data_summary = forecast        
        prompt = f"""Analyze the following dataset:

Summary Statistics:
{data_summary}

Now, based on this data, {prompt}

Provide a statistical analysis, including exact counts and percentages where applicable."""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                 {"role": "system", "content": System_Prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response['choices'][0]['message']['content']
        
    except Exception as e:
        st.error(f"Error in generating NLG response: {str(e)}")
        return "Sorry, I couldn't generate a response at this time."

def forecast_sales(data, sales_column):
    sales_data = data[sales_column].tolist()
    sales_data_str = ', '.join(map(str, sales_data))

    prompt = f"Given the following sales data: {sales_data_str}, forecast the next 12 periods of sales. Return only the forecasted sales values as a comma-separated string."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.1,
        messages=[
            {"role": "system", "content": System_Prompt},
            {"role": "user", "content": prompt}
        ]
    )

    forecasted_values = response['choices'][0]['message']['content']
    
    print("API Response:", forecasted_values)
    
    try:
        forecasted_data = [float(value) for value in forecasted_values.split(',')]
    except ValueError as e:
        st.error("Error parsing forecasted values. Please check the API response.")
        print("Error:", e)
        return None

    return forecasted_data

def generate_explanation(data, forecast):
    if forecast is None:
        return "Forecast data is unavailable."
    forecast_str = ', '.join(map(str, forecast))
    # Continue with other logic
    return f"The forecasted values are: {forecast_str}"
    
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
    {System_Prompt_Forecast}
    
    1. Analyze the provided data and identify trends, anomalies, and patterns. {historical_data_str}

    2. Based on the provided data, describe the forecasted sales values for the next 12 periods. {forecast_str}

    3. Use context to enhance the insights and analysis. {context}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.7,
        max_tokens=500,
        messages=[
            {"role": "system", "content": System_Prompt_Forecast},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']

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
    
    data_input_method = st.selectbox("Upload Sales Data Here (CSV only) or Manually Input Sales Data", ["CSV", "Manual Data"])

    if data_input_method == "CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Sales Data: First Rows", data.head())
            st.write("Sales Data: Last Rows", data.tail())
            sales_column = st.selectbox("Select the (Sales or Revenue) column to forecast:", data.columns)
    else:
        st.write("Enter your sales data below:")
        sales_data = st.text_area("Sales Data (comma-separated, e.g., 100, 150, 200)", "")
        if sales_data:
            sales_list = [float(x) for x in sales_data.split(",")]
            data = pd.DataFrame({'Sales': sales_list})
            sales_column = 'Sales' 

    if 'data' in locals() and 'sales_column' in locals():
        if st.button("Forecast Sales"):
            forecast = forecast_sales(data, sales_column)
            st.write("Forecasted Sales:", forecast)

            # Visualization
            st.header("Forecast Sales Chart")
            st.line_chart(forecast)

            #NLG
            st.header("Summary of Statistical Report")
            prompt = f"""
            {System_Prompt_Forecast}
            Provided with the sales data, give the forecast for the next 12 periods. Provide the statistical analysis, trends, insights, and conclusion.
            """
            nlg_response = generate_nlg_response(prompt, forecast)
            st.write("Forecast Sales:", nlg_response)

            #Analysis with RAG
            st.header("Summary of Sales Analyses")
            explanation = generate_explanation(data, forecast)
            st.write("Explanation:", explanation)
