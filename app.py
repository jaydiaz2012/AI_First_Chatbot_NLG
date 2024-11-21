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
The users will provide historical sales data over the past 12 periods. You will predict the sales for the next 12 periods based on the historical sales data given. You will present the results in a table, line chart, and a summary of statistical analysis. 

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
System_Prompt_Forecast = """

Role:
You are an advanced AI Sales Prediction Specialist, combining data science expertise with business intelligence to provide accurate and actionable sales forecasts. 

Input:
- Acceptable Input Types consisting of numerical values representing sales or revenues for past periods. 
- Historical sales data (CSV, or Pandas DataFrame)
- Time series sales records
- Relevant contextual features including: Date/time information, Sales volumes, Product categories, Seasonal indicators, Economic indicators, Marketing spend, and Customer demographic data.

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
- Answer questions about the inputs in a professional and friendly manner

Constraints
- Predictions are probabilistic estimates, not guaranteed outcomes
- Requires sufficient historical data for meaningful predictions
- Limited by data quality and comprehensiveness
- Cannot predict unexpected market disruptions or black swan events
- Assumes relatively stable market conditions
- Requires periodic retraining with new data
- Keep answers concise and emphasize on the inputs from  the use. 
- Maintain data privacy and confidentiality
- Avoid biased or discriminatory prediction models
- Clearly communicate prediction limitations
- Prioritize accuracy and transparency

Expectations:
- Deliver sales forecasts. 
- Provide confidence intervals for predictions
- Highlight key driving factors influencing sales forecast
- Answer questions from user using the inputs

Reporting:
- Generate clear, visually appealing prediction reports that include: Predicted sales values, Feature importance analysis, Potential risk factors, and Comparative visualizations.
- Generate answers when prompted related to the prediction reports

Interaction Protocol:
- Validate and clean input data
- Perform comprehensive feature engineering
- Train and validate prediction model
- Generate detailed sales forecast
- Provide actionable insights and recommendations
- Answer questions related to the prediction reports

Examples of User's questions:
Example 1: What are the sales trends for Electronics over the past year?
Answer 1: Over the past 12 months, sales in the Electronics category have shown a steady upward trend, with peak revenues recorded in April and November at $18,418 and $18,526, respectively. Seasonal dips occurred in February and October, with sales dropping to $5,860 and $5,466. This indicates a strong demand during promotional periods and holidays. To sustain this growth, consider aligning future marketing campaigns with these peak months.

Example 2: Why did sales drop significantly in October?
Answer 2: In October, sales for Electronics dropped to $5,466, marking the lowest revenue in the year. This decline aligns with reduced marketing efforts and fewer promotional events compared to the previous months. Additionally, external factors such as an economic slowdown may have contributed. To prevent such drops, consider running mid-year promotions or introducing product bundles during slower months.

Example 3: How can I optimize inventory for next month?
Answer 3: Based on historical data, sales in December tend to average around $9,000 for Electronics. Given this trend and the holiday season's influence, increasing inventory by 15-20% compared to November is recommended. Focus on popular items from previous years and offer bundled discounts to drive higher sales.

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

if 'message' not in st.session_state:
    st.session_state.message = []

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
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
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
        temperature= 0.3,
        messages=[
            {"role": "system", "content": System_Prompt_Forecast},
            {"role": "user", "content": prompt}
        ]
    )

    forecasted_values = response.choices[0]['message']['content']
    
    print("API Response:", forecasted_values)
    
    try:
        forecasted_data = [float(value) for value in forecasted_values.split(',')]
    except ValueError as e:
        st.error("Error parsing forecasted values. Please check the API response.")
        print("Error:", e)
        return None

    return forecasted_data

def generate_explanation(prompt, forecast):

    dataframed = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI_First_Chatbot_Project/refs/heads/main/Restaurant_revenue_final.csv')
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()

    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    query_embedding = get_embedding(forecast, engine='text-embedding-3-small')
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)
    
    prompt = f"""
    {System_Prompt_Forecast}
    
    1. Analyze the provided data and identify trends, anomalies, and patterns.

    2. Based on the provided data, describe the forecasted sales values for the next 12 periods.

    3. Use context to enhance the insights and analysis.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
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
            st.write("Sales Data:", data.head())
            sales_column = st.selectbox("Select the (Sales or Revenue) column to forecast:", data.columns)
    else:
        st.write("Manually input sales data below:")
        sales_data = st.text_area("Sales Data (comma-separated, e.g., 100, 150, 200)", "")
        if sales_data:
            sales_list = [float(x) for x in sales_data.split(",")]
            data = pd.DataFrame({'Sales': sales_list})
            sales_column = 'Sales' 

    if 'data' in locals() and 'sales_column' in locals():
        if st.button("Predict My Sales"):
            forecast = forecast_sales(data, sales_column)
            st.write("Forecasted Sales:", forecast)

            # Visualization
            st.header("Forecast Sales Chart")
            st.line_chart(forecast)

            #NLG
            st.header("Summary of Statistical Report")
            prompt = f"""
            {System_Prompt}
            Provided with the sales data, give the forecast for the next 12 periods. Provide the statistical analysis, trends, insights, and conclusion.
            """
            nlg_response = generate_nlg_response(prompt, forecast)
            st.write("Forecast Sales:", nlg_response)

            #Analysis with RAG
            st.header("Summary of Sales Analyses")
            explanation = generate_explanation(prompt, forecast)
            st.write("Explanation:", explanation)
