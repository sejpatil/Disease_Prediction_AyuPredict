

import streamlit as st
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import joblib 
import pickle
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

# Assuming you have a list of disease labels
disease_labels = ["Drug Reaction", "Malaria", "Allergy", "Hypothyroidism", "Psoriasis",
    "GERD", "Chronic cholestasis", "hepatitis A", "Osteoarthristis",
    "(vertigo) Paroymsal  Positional Vertigo", "Hypoglycemia", "Acne",
    "Diabetes", "Impetigo", "Hypertension", "Peptic ulcer diseae",
    "Dimorphic hemorrhoids(piles)", "Common Cold", "Chicken pox",
    "Cervical spondylosis", "Hyperthyroidism", "Urinary tract infection",
    "Varicose veins", "AIDS", "Paralysis (brain hemorrhage)", "Typhoid",
    "Hepatitis B", "Fungal infection", "Hepatitis C", "Migraine",
    "Bronchial Asthma", "Alcoholic hepatitis", "Jaundice", "Hepatitis E",
    "Dengue", "Hepatitis D", "Heart attack", "Pneumonia", "Arthritis",
    "Gastroenteritis", "Tuberculosis"]

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([disease_labels])





df1 = pd.read_csv(r"Symptom-severity.csv")
df2 = pd.read_csv(r"symptom_Description.csv")
df3 = pd.read_csv(r"symptom_precaution_new-1.csv")
df4 = pd.read_csv(r"symptom_precaution.csv")
df5 = pd.read_csv(r"Symptom-severity.csv")




#load models
with open('disease_predict.pkl', 'rb') as model_file:
    disease_predict = pickle.load(model_file)
with open('hospital_suggest.pkl', 'rb') as model_file:
    hospital_suggest = pickle.load(model_file)



    

def search_hospitals(city, hospital_suggest):
    if city in hospital_suggest:
        hospitals = hospital_suggest[city]
        from sklearn.neighbors import NearestNeighbors

        # Print the nearest hospitals
        st.write("Hospitals in {}:".format(city))
        for i, hospital in enumerate(hospitals):
            st.write("{}. {}".format(i+1, hospital))
    else:
        st.write("No hospital suggestions available for {}".format(city))

def predd(disease_predict,selected_symptoms):
    psymptoms = selected_symptoms

    #print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]
    psy = [psymptoms]
    pred2 = disease_predict.predict(psy)
    return pred2
    # disp= df2[df2['Disease']==pred2[0]]
    # disp = disp.values[0][1]
    # recomnd = df3[df3['Disease']==pred2[0]]
    # c=np.where(df3['Disease']==pred2[0])[0][0]
    # precuation_list=[]
    # for i in range(1,len(df3.iloc[c])):
    #       precuation_list.append(df3.iloc[c,i])
    # print("The Disease Name: ",pred2[0])
    # print("The Disease Discription: ",disp)
    # print("Recommended Things to do at home: ")
    # for i in precuation_list:
    #     print(i)

# def predict_disease(selected_symptoms, disease_predict, mlb, symptoms):
#     # Create a binary array representing symptoms
#     symptom_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
#     # Use the model to predict the disease
#     predicted_labels = disease_predict.predict([symptom_vector])
#     # Convert predicted labels back to disease names using the MultiLabelBinarizer
#     predicted_diseases = mlb.inverse_transform(predicted_labels)
#     return predicted_diseases

    
     
def main():
    st.sidebar.title("Login to AyuPredict")
    st.sidebar.text_input("Email ID:")
    st.sidebar.text_input("Current City")
    st.sidebar.button("Submit")
    st.title("AyurPredict - A Disease Prediction Model ")
    st.image(r"Sukh Sutra Ayurveda_.jpg", width=500)
    st.header(" A Disease Prediction Model")
    st.subheader("This Disease Prediction Model can predict your disease and and provide you an ayurvedic treatment")
    st.write("In the digital age, Ayurvedify emerges as a beacon, illuminating the path towards personalized and accessible healthcare solutions. This project explores the synthesis of Ayurvedic wisdom with ML algorithms, culminating in the creation of a dynamic, user-friendly platform.")

    st.radio("Select Gender",["Male","Female","Other"])
    st.number_input("Enter age", 0,80)
    st.write("Symptoms:")

    symptoms=['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips', 'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness', 'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell ofurine', 'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain', 'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf', 'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze', 'prognosis']
    selected_symptoms = st.multiselect("Select symptoms:", symptoms)

    # Display the selected symptoms with their appropriate indices
    st.write("Selected symptoms:")
    index_list=[]
    for symptom in selected_symptoms:
        index = symptoms.index(symptom) 
        #index_list.append(index)# Get the index of the symptom in the original list
        st.write(f"{index}") 
        
    symptom_list = selected_symptoms + ['0'] * (17 - len(selected_symptoms))
            
    button_clicked = st.button("Predict")
    if selected_symptoms and button_clicked:
            prediction = predd(disease_predict,symptom_list)
            st.write("The predicted disease is: ", prediction)
            
            
        

    st.write("Choose your city:")   
    selected_city = st.multiselect("Choose your symptoms",["Ahmedabad", "Ajmer", "Aligarh", "Amritsar", "Ananthapur", "Bangalore", "Bareilly", "Belgaum", "Bellary", "Belthangady", "Bhiwandi", "Bhopal", "Bijnor", "Buldana", "Chandigarh", "Chellakere", "Chennai", "Cochin", "Coimbatore", "Cuddalore", "Cuttack", "Delhi", "Dibrugarh", "Eastgodavari", "Eluru", "Ernakulam", "Erode", "Faridabad", "Gandhinagar", "Ghaziabad", "Guntur", "Gurgaon", "Guwahati", "Himayatnagar", "Hosur", "Hyderabad", "Indore", "Jaipur", "Jammu", "Kanpur", "Karur", "Karwar", "Khurda", "Kolar", "Kolkata", "Koparkhairane", "Korba", "Kota", "Kozhicode", "Kundapur", "Kunnamkulam", "Kurnool", "Kurukshetra", "Kutch", "Lucknow", "Madurai", "Malapuram", "Manacaud", "Mandya", "Mangalore", "Margao", "Meerut", "Mehsana", "Miryalaguda", "Mohali", "Muktsar", "Mumbai", "Mysore", "Nagercoil", "Nagpur", "Nellore", "New Delhi", "Palakkad", "Panchkula", "Patan", "Patna", "Pondicherry", "Pune", "Raigad", "Raipur", "Rajamundry", "Rajkot", "Ramanagaram", "Ranchi", "Salem", "Sangli", "Secunderabad", "Shamshabad", "Sindhudurga", "Sriganganagar", "Srinagar", "Surat", "Tenali", "Thane", "Tirunelveli", "Trivandrum", "Tumkur", "Tuticorin", "Udupi", "Ulhasnagar", "Vaishali", "Varanasi", "Visakhapatnam", "Vizag", "Warangal", "Wardha"])
    clicked= st.button("Search Hospitals")
    if selected_city:
        if clicked:
            for city in selected_city:
                search_hospitals(city, hospital_suggest)
    
        
    
    
    
if __name__ == '__main__':
    main()