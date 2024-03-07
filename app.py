
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from mcqgenerator.utils import read_file,get_table_data
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from mcqgenerator.MCQGenerator import generate_evaluate_chain
from mcqgenerator.logger import logging

with open('Respons.json','r') as file:
    RESPONSE_JSON = json.load(file)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
st.image('images.jpg', caption='Image credit Steve Johnson')
st.title(''' Multiple Choice Question Generator''')
st.markdown("""
<div style="background-color:#DFE9F5;padding:10px;border-radius:10px;text-align:justify;">
<p style="text-align:left;color:green;">This system is designed to create multiple-choice questions using Generative AI technology, particularly utilizing LangChain and the OpenAI API.
The tool allows users to upload text or PDF files by drag and drop them into the provided area.
Once the file is uploaded, users can specify the subject of the content in the file and adjust the difficulty level of the questions to be generated. Essentially,
it's a platform that automates the process of generating multiple-choice questions based on the content of a given text or PDF file, with customization options for subject and difficulty level.
<br><span style="color:red;">Note : This model's maximum data context length is 16385 tokens.</span></p>
</div>
""", unsafe_allow_html=True)
upload_file = st.file_uploader('Please upload your pdf or txt file here')
mcq_count = st.number_input('Number of question',min_value=3,
                            max_value=50,
                            value=3,placeholder='3')
subject = st.text_input('Insert Subject',
                        placeholder='example : Machine Learning', 
                        max_chars=30)

tone = st.text_input('Insert difficulty level', 
                        max_chars=30,
                        placeholder='easy | medium | hard')

button = st.button('create')

if button and upload_file is not None and mcq_count and subject and tone :
    with st.spinner('Loading...'):
        try:
            text = read_file(upload_file)

            with get_openai_callback() as cb:
                response = generate_evaluate_chain.invoke(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject":subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                    )
            st.write(f"Total Tokens : {cb.total_tokens}")
            st.write(f"Prompt Tokens : {cb.prompt_tokens}")
            st.write(f"Completion Tokens : {cb.completion_tokens}")
            st.write(f"Total Cost : {cb.total_cost}")
        
        except Exception as e :
            st.error(f'{type(e)} : {e}')
        
        else:
            if isinstance(response,dict):
                quiz = response.get('quiz',None)
                quiz2 = response.get('review',None)
                            
            if quiz is not None:
                table_data = get_table_data(quiz)
                table_data2 = get_table_data(quiz2)
                if table_data is not None:
                    df = pd.DataFrame(table_data)
                    df.index = df.index+1
                    st.write('First Option')
                    st.table(df)
                    csv = convert_df(df)
                    
                    #download button
                    st.download_button(
                        label="Download First Option",
                        data=csv,
                        file_name='quiz_1.csv',
                        mime='text/csv')
                    
                    df2 = pd.DataFrame(table_data2)
                    df2.index = df2.index+1
                    st.write('Second Option')
                    st.table(df2)
                    
                    #download button
                    st.download_button(
                        label="Download Second Option",
                        data=csv,
                        file_name='quiz_2.csv',
                        mime='text/csv')
                else:
                    st.error('Error in the table data')
            
            else:
                st.write(response)
        
    st.success('Done!')

