import pandas as pd
import os
import openai
from openai import OpenAI
from openai import AzureOpenAI
import streamlit as st
import PyPDF2

model = "gpt-4-turbo" # Specify the desired model

endpoint = os.getenv("ENDPOINT_URL", "https://ansysaceaiservicegpteastus2.openai.azure.com/")

deployment = os.getenv("DEPLOYMENT_NAME", model)

subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "349d19457eee42f2910049050014fdd9")

api_version="2024-08-01-preview"

#Initialize Azure OpenAI-client

client = AzureOpenAI(

azure_endpoint=endpoint,

api_key=subscription_key,

api_version=api_version,)
client = OpenAI(api_key='349d19457eee42f2910049050014fdd9')



# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to interact with GPT-4 Turbo
def process_prompt_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts tabular data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
def main():
    st.title("Tabular Data Extractor with GPT")
    st.write(
        "Upload a PDF file containing tabular data, and this app will extract the tabular information."
    )

    # File upload
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("Processing your file...")

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("PDF content extracted successfully!")

        # Generate GPT prompt
        custom_prompt = (
            "Extract the table information from the following text:\n\n" + pdf_text
        )

        # Get tabular data from GPT
        st.write("Extracting tabular data using GPT...")
        extracted_data = process_prompt_with_gpt(custom_prompt)

        st.write("Here is the extracted tabular data:")
        st.text_area("Tabular Data Output", extracted_data, height=300)

        # Convert to DataFrame if possible
        try:
            df = pd.read_csv(pd.compat.StringIO(extracted_data))
            st.write("Tabular data as a DataFrame:")
            st.dataframe(df)
        except Exception:
            st.write("Unable to convert the output into a DataFrame.")

if __name__ == "_main_":
    main()