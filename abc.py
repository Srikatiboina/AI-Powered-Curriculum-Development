# **1. Importing Necessary Libraries** 📚
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from db import *
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = API_KEY

# **2. Load the trained model**
with open("weights.pkl", "rb") as pickleFile:
    regressor = pickle.load(pickleFile)

# **3. Loading Dataset**
df = pd.read_csv("C:/Users/Asfiy/OneDrive/Desktop/Robokalam Intership Tasks/python-venv/mldata.csv")
df['workshops'] = df['workshops'].replace(['testing'], 'Testing')

n = df['Suggested Job Role'].unique()
print(len(n))

print(f'The shape of our training set: {df.shape[0]} professionals and {df.shape[1]} features')

# **5. Feature Engineering**
# (a) Binary Encoding for Categorical Variables
binary_cols = ["self-learning capability?", "Extra-courses did", "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

print("\n\nList of Categorical features: \n", df.select_dtypes(include=['object']).columns.tolist())

# (b) Number Encoding for Categorical Variables
num_cols = ["reading and writing skills", "memory capability score"]
for col in num_cols:
    df[col] = df[col].map({"poor": 0, "medium": 1, "excellent": 2})

category_cols = ['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']
for col in category_cols:
    df[col + "_code"] = df[col].astype('category').cat.codes

print("\n\nList of Categorical features: \n", df.select_dtypes(include=['object']).columns.tolist())

# (c) Dummy Variable Encoding
df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])

print("List of Numerical features: \n", df.select_dtypes(include=np.number).columns.tolist())

# Prepare lookup dictionaries for categorical features
C = dict(zip(df['certifications'].unique(), df['certifications_code'].unique()))
W = dict(zip(df['workshops'].unique(), df['workshops_code'].unique()))
ISC = dict(zip(df['Interested subjects'].unique(), df['Interested subjects_code'].unique()))
ICA = dict(zip(df['interested career area '].unique(), df['interested career area _code'].unique()))
TOCO = dict(zip(df['Type of company want to settle in?'].unique(), df['Type of company want to settle in?_code'].unique()))
IB = dict(zip(df['Interested Type of Books'].unique(), df['Interested Type of Books_code'].unique()))
Range_dict = {"poor": 0, "medium": 1, "excellent": 2}

# Define the function to preprocess input data and predict
def inputlist(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points, self_learning_capability, Extra_courses_did, Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, reading_and_writing_skills, memory_capability_score, smart_or_hard_work, Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area):
    try:
        # Feature list
        Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]
        input_list_col = [
            self_learning_capability,
            Extra_courses_did,
            Taken_inputs_from_seniors_or_elders,
            worked_in_teams_ever,
            Introvert,
            reading_and_writing_skills,
            memory_capability_score,
            smart_or_hard_work,
            Management_or_Techinical,
            Interested_subjects,
            Interested_Type_of_Books,
            certifications,
            workshops,
            Type_of_company_want_to_settle_in,
            interested_career_area
        ]
        feed = []
        
        for i in input_list_col:
            if i in ["Yes", "No"]:
                feed.append(1 if i == "Yes" else 0)
            elif i in ["Smart worker", "Hard Worker", "Management", "Technical"]:
                feed.extend([1, 0] if i in ["Smart worker", "Management"] else [0, 1])
            else:
                for key, val in {**Range_dict, **C, **W, **ISC, **ICA, **TOCO, **IB}.items():
                    if i == key:
                        feed.append(val)
                        break
        
        # Combine all features
        t = Afeed + feed
        output = regressor.predict([t])
        
        return output
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit App
def main():
    st.markdown("""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1>👨🏻‍💻 Prototype for course selecting according to the interest of user 👨🏻‍💻</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Career _Isometric.png")
    with col2:
        st.image("career.png")
    with col3:
        st.image("Career _Outline.png")

    st.markdown("""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Your Friendly Career Advisor<h2>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Your Information")
    Name = st.sidebar.text_input("Full Name")
    Contact_Number = st.sidebar.text_input("Contact Number")
    Email_address = st.sidebar.text_input("Email address")

    if not Name or not Email_address:
        st.sidebar.warning("Please fill out your name and EmailID")

    if Name and Contact_Number and Email_address:
        st.sidebar.success("Thanks!")

    Logical_quotient_rating = st.slider('Rate your Logical quotient Skills', 0, 10, 1)
    coding_skills_rating = st.slider('Rate your Coding Skills', 0, 10, 1)
    hackathons = st.slider('Enter number of Hackathons participated', 0, 10, 1)
    public_speaking_points = st.slider('Rate Your Public Speaking', 0, 10, 1)
    self_learning_capability = st.selectbox('Self Learning Capability', ('Yes', 'No'))
    Extra_courses_did = st.selectbox('Extra courses', ('Yes', 'No'))
    Taken_inputs_from_seniors_or_elders = st.selectbox('Took advice from seniors or elders', ('Yes', 'No'))
    worked_in_teams_ever = st.selectbox('Team Co-ordination Skill', ('Yes', 'No'))
    Introvert = st.selectbox('Introvert', ('Yes', 'No'))
    reading_and_writing_skills = st.selectbox('Reading and writing skills', ('poor', 'medium', 'excellent'))
    memory_capability_score = st.selectbox('Memory capability score', ('poor', 'medium', 'excellent'))
    smart_or_hard_work = st.selectbox('Smart or Hard Work', ('Smart worker', 'Hard Worker'))
    Management_or_Techinical = st.selectbox('Management or Technical', ('Management', 'Technical'))
    Interested_subjects = st.selectbox('Interested Subjects', ('programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking'))
    Interested_Type_of_Books = st.selectbox('Interested Books Category', ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'))
    certifications = st.selectbox('Certifications', ('information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python'))
    workshops = st.selectbox('Workshops Attended', ('Testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies'))
    Type_of_company_want_to_settle_in = st.selectbox('Type of Company You Want to Settle In', ('BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based'))
    interested_career_area = st.selectbox('Interested Career Area', ('testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing'))
    # Placeholder for the new curriculum generation section
    curriculum_section = st.empty()

    with curriculum_section.container():
        st.markdown("## Generate Curriculum")
        
        st.markdown("**Select LLM**")
        llm_selection = st.selectbox("Select LLM", ("gpt-3.5-turbo", "gpt-4"))
        
        st.markdown("**Select levels which you want to learn**")
        levels_selection = st.selectbox("Select Level", ("Primary school(Level1)", "Secondary school(Level2)", "High school(Level3)", "Undergraduate(Level4)", "Graduate(Level5)"))
        
        st.markdown("**Select number of weeks**")
        weeks_selection = st.slider("Number of weeks", 1, 52, 10)

        generate_button = st.button("Generate Curriculum")
        
        if generate_button:
            with st.spinner("Generating curriculum..."):
                # Call the function to generate the curriculum here
                time.sleep(2)  # Simulate a delay for generating the curriculum
                st.success("Curriculum generated successfully!")
                st.markdown("### Your generated curriculum will appear here.")

    if st.button("Submit"):
        result = inputlist(
            Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, hackathons, 
            public_speaking_points, self_learning_capability, Extra_courses_did, Taken_inputs_from_seniors_or_elders, 
            worked_in_teams_ever, Introvert, reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
            Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, certifications, workshops, 
            Type_of_company_want_to_settle_in, interested_career_area
        )
        if result:
            st.write(f"The suggested job role is: {result[0]}")
        else:
            st.write("An error occurred. Please try again.")

if __name__ == '__main__':
    main()
