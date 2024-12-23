# **1. Importing Necessary Libraries** 📚
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from db import *
from predict import clf1

# Load environment variables
load_dotenv()

# **2. Load the trained model**
with open("weights.pkl", "rb") as pickleFile:
    regressor = pickle.load(pickleFile)

# **3. Load Dataset**
df = pd.read_csv("C:/Users/Asfiy/OneDrive/Desktop/My Learnings/My Internships/Robokalam Intership Tasks/python-venv/mldata.csv")

# Data Cleaning
df['workshops'] = df['workshops'].replace(['testing'], 'Testing')

# Print unique job roles and dataset shape
n = df['Suggested Job Role'].unique()
print(len(n))
print(f'The shape of our training set: {df.shape[0]} professionals and {df.shape[1]} features')

# **5. Feature Engineering**
# (a) Binary Encoding for Categorical Variables
binary_cols = ["self-learning capability?", "Extra-courses did", 
               "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# (b) Number Encoding for Categorical Variables
num_cols = ["reading and writing skills", "memory capability score"]
for col in num_cols:
    df[col] = df[col].map({"poor": 0, "medium": 1, "excellent": 2})

# (c) Dummy Variable Encoding
category_cols = ['certifications', 'workshops', 'Interested subjects', 
                 'interested career area ', 'Type of company want to settle in?', 
                 'Interested Type of Books']
for col in category_cols:
    df[col + "_code"] = df[col].astype('category').cat.codes

# Prepare lookup dictionaries for categorical features
C = dict(zip(df['certifications'].unique(), df['certifications_code'].unique()))
W = dict(zip(df['workshops'].unique(), df['workshops_code'].unique()))
ISC = dict(zip(df['Interested subjects'].unique(), df['Interested subjects_code'].unique()))
ICA = dict(zip(df['interested career area '].unique(), df['interested career area _code'].unique()))
TOCO = dict(zip(df['Type of company want to settle in?'].unique(), df['Type of company want to settle in?_code'].unique()))
IB = dict(zip(df['Interested Type of Books'].unique(), df['Interested Type of Books_code'].unique()))
Range_dict = {"poor": 0, "medium": 1, "excellent": 2}

# Define the function to preprocess input data and predict
def inputlist(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, 
               hackathons, public_speaking_points, self_learning_capability, Extra_courses_did, 
               Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, 
               reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
               Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, 
               certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area):
    
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
    
    # Combine all features and predict
    t = Afeed + feed
    output = clf1.predict([t])
    
    return output

# Streamlit App
def main():
    # HTML headers
    html1 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1>👨🏻‍💻 Prototype for course selecting according to the interest of user 👨🏻‍💻</h1>
    </div>
    """
    st.markdown(html1, unsafe_allow_html=True)

    # Display images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Career _Isometric.png")
    with col2:
        st.image("career.png")
    with col3:
        st.image("Career _Outline.png")

    # Subheading
    html2 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Your Friendly Career Advisor<h2>
    </div>
    """
    st.markdown(html2, unsafe_allow_html=True)

    # Sidebar for user input
    st.sidebar.title("Your Information")
    Name = st.sidebar.text_input("Full Name")
    Contact_Number = st.sidebar.text_input("Contact Number")
    Email_address = st.sidebar.text_input("Email address")

    if not Name and not Email_address:
        st.sidebar.warning("Please fill out your name and Email ID")

    if Name and Contact_Number and Email_address:
        st.sidebar.success("Thanks!")

    # User input fields
    Logical_quotient_rating = st.slider('Rate your Logical quotient Skills', 0, 10, 1)
    coding_skills_rating = st.slider('Rate your Coding Skills', 0, 10, 1)
    hackathons = st.slider('Enter number of Hackathons participated', 0, 10, 1)
    public_speaking_points = st.slider('Rate Your Public Speaking', 0, 10, 1)
    
    # Select boxes for categorical data
    self_learning_capability = st.selectbox('Self Learning Capability', ('Yes', 'No'))
    Extra_courses_did = st.selectbox('Extra courses', ('Yes', 'No'))
    Taken_inputs_from_seniors_or_elders = st.selectbox('Took advice from seniors or elders', ('Yes', 'No'))
    worked_in_teams_ever = st.selectbox('Team Co-ordination Skill', ('Yes', 'No'))
    Introvert = st.selectbox('Introvert', ('Yes', 'No'))
    reading_and_writing_skills = st.selectbox('Reading and writing skills', ('poor', 'medium', 'excellent'))
    memory_capability_score = st.selectbox('Memory capability score', ('poor', 'medium', 'excellent'))
    smart_or_hard_work = st.selectbox('Smart or Hard Work', ('Smart worker', 'Hard Worker'))
    Management_or_Techinical = st.selectbox('Management or Technical', ('Management', 'Technical'))
    Interested_subjects = st.selectbox('Interested Subjects', ('programming', 'Management', 'data engineering', 
                                                              'networks', 'Software Engineering', 
                                                              'cloud computing', 'parallel computing', 'IOT', 
                                                              'Computer Architecture', 'hacking'))
    Interested_Type_of_Books = st.selectbox('Interested Books Category', 
                                             ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 
                                              'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 
                                              'Art', 'Encyclopedias', 'Religion-Spirituality', 
                                              'Action and Adventure', 'Comics', 'Horror', 'Satire', 
                                              'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 
                                              'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 
                                              'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'))
    certifications = st.selectbox('Certifications', ('information security', 'shell programming', 
                                                      'r programming', 'distro making', 'machine learning', 
                                                      'full stack', 'hadoop', 'app development', 'python'))
    workshops = st.selectbox('Workshops Attended', ('Testing', 'database security', 'game development', 
                                                     'data science', 'system designing', 'hacking', 
                                                     'cloud computing', 'web technologies'))
    Type_of_company_want_to_settle_in = st.selectbox('Type of Company You Want to Settle In', 
                                                      ('BPA', 'Cloud Services', 'product development', 
                                                       'Testing and Maintenance Services', 'SAaS services', 
                                                       'Web Services', 'Finance', 'Sales and Marketing', 
                                                       'Product based', 'Service Based'))
    interested_career_area = st.selectbox('Interested Career Area', 
                                            ('testing', 'system developer', 'Business process analyst', 
                                             'security', 'developer', 'cloud computing'))

    # Prediction button
    if st.button('Predict'):
        output = inputlist(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, 
                           hackathons, public_speaking_points, self_learning_capability, Extra_courses_did, 
                           Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, 
                           reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
                           Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, 
                           certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area)

        # Display the prediction result
        st.success(f'You might want to learn about: {output[0]}')
        html3 = """
        <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
          <h3>--CLICK ON GENERATE CURRICULUM BUTTON TO CONTINUE--</h3>
        </div>
        """
        st.markdown(html3, unsafe_allow_html=True)

        # Curriculum Generation
        from config import LLM_CONFIGURATIONS
        selected_llm = st.selectbox("Select LLM", LLM_CONFIGURATIONS.keys())
        level = st.selectbox("Select levels you want to learn", 
                             ["Primary school(Level 1)", "Secondary school(Level 2)", 
                              "University(Level 3)", "Working Professional(Level 4)"])
        no_of_weeks = st.number_input("Select number of weeks", min_value=1, max_value=100, value=10)

        # Prompts for OpenAI
        system_prompt = "You are an assistant responsible for generating structured curriculum based on user inputs. Generate response in JSON format."
        user_prompt = f"""Generate the curriculum for 
        subject: {output[0]},
        level: {level},
        no_of_weeks: {no_of_weeks}
        """

        # Generate Curriculum Button
        if st.button("Generate curriculum"):
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            openai_response = openai_client.chat.completions.create(
                model=LLM_CONFIGURATIONS[selected_llm]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result = openai_response.choices[0].message.content
            st.success(result)

# Entry point for the application
if __name__ == '__main__':
    main()
