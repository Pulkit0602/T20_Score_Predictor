import streamlit as st
import numpy as np
import pandas as pd
import pickle

#READING FILES
X = pickle.load(open('dataset_level2.pkl','rb'))  
batting_team = X["batting_team"].unique() 
bowling_team = X["bowling_team"].unique()
city = X["city"].unique()

#LOADING MODEL
model1= pickle.load(open('RandomForestRegressor.pkl','rb'))  

#DEVELOPING INTERFACE
st.title("T20 Score Predictor")

#City , Bowling & Batting Team
col1,col2,col3= st.columns(3)
with col1:    
    batting_team = st.selectbox("Batting Team" , batting_team)

with col2:
    bowling_team = st.selectbox("Bowling Team" , bowling_team)

with col3:
    city = st.selectbox("City" , city)

#Wicket, CurrentRun,Overs Completed
col1,col2,col3 = st.columns(3)
with col1:    
    score = st.number_input('Score')

with col2:
    overs = st.number_input('Overs (Note >5')

with col3:
    wickets = st.number_input('wickets')

last_five = st.number_input('Last 5 Overs Score')

if st.button("Predict"):
    over , ball = str(overs).split(".")
    balls_left = 120 - (int(over)*6 + int(ball))
    player_left = 10 - wickets
    run_rate = score/overs


    result = model1.predict(pd.DataFrame(columns=['batting_team', 'bowling_team', 'city', 'current_score', 'Balls_Left', "player_left", "Run_Rate", "last_five" ],
                                data=np.array([batting_team, bowling_team, city, score, balls_left, player_left, run_rate, last_five ]).reshape(1,8)))
                            
    st.header("Predicted Score: " + str(np.ceil(result[0])))