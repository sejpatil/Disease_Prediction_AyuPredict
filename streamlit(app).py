import streamlit as st
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib 
import pickle
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

df1 = pd.read_csv(r"C:\Users\Sejal\OneDrive\Desktop\Disease_Datset_DP\dataset.csv")
df2 = pd.read_csv(r"C:\Users\Sejal\OneDrive\Desktop\Disease_Datset_DP\symptom_Description.csv")
df3 = pd.read_csv(r"C:\Users\Sejal\OneDrive\Desktop\Disease_Datset_DP\symptom_precaution_new-1.csv")
df4 = pd.read_csv(r"C:\Users\Sejal\OneDrive\Desktop\Disease_Datset_DP\symptom_precaution.csv")
df5 = pd.read_csv(r"C:\Users\Sejal\OneDrive\Desktop\Disease_Datset_DP\Symptom-severity.csv")




#load models
with open('disease_predict.pkl', 'rb') as model_file:
    disease_predict = pickle.load(model_file)
with open('hospital_suggest.pkl', 'rb') as model_file:
    hospital_suggest = pickle.load(model_file)




def predict_disease(symptoms, disease_predict):
    # Dictionary mapping symptoms to indices
    symptom_to_index = {symptom: index for index, symptom in enumerate(symptoms)}
    # Convert selected symptoms into a list of indices
    selected_symptoms_indices = [symptom_to_index[symptom] for symptom in selected_symptoms_indices]
    predicted_disease = disease_predict.predict(selected_symptoms_indices)  # Assuming predict method is available in your model
    # Return the predicted disease
    return predicted_disease
    # Example usage:
    # Load your machine learning model
# Assuming selected_symptoms_indices is the list of indices obtained from selected symptoms
# Call predict_disease function to get the predicted disease

    # # Encode symptoms using one-hot encoding
    # mlb = MultiLabelBinarizer()
    # symptoms_encoded = mlb.fit_transform([symptoms])

    # # Ensure that the encoded symptoms have the same number of features as the model expects
    # if symptoms_encoded.shape[1] != 17:
    #     # If the number of features is not 17, pad with zeros to match the expected number of features
    #     num_missing_features = 17 - symptoms_encoded.shape[1]
    #     symptoms_encoded = np.pad(symptoms_encoded, ((0, 0), (0, num_missing_features)), mode='constant')

    # # Predict disease
    # prediction = disease_predict.predict(symptoms_encoded)
    # return prediction
    

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

    
    
    
     
def main():
    st.sidebar.title("Login to Ayurvedify")
    st.sidebar.text_input("Email ID:")
    st.sidebar.text_input("Current City")
    st.sidebar.button("Submit")
    st.title("Ayurvedify")
    st.image(r"Sukh Sutra Ayurveda_.jpg", width=500)
    st.header("A Healthcare Bot")
    st.subheader("This healthcare bot can predict your disease and and provide you an ayurvedic treatment")
    st.write("In the digital age, Ayurvedify emerges as a beacon, illuminating the path towards personalized and accessible healthcare solutions. This project explores the synthesis of Ayurvedic wisdom with ML algorithms, culminating in the creation of a dynamic, user-friendly platform.")

    st.radio("Select Gender",["Male","Female","Other"])
    st.number_input("Enter age", 0,80)
    st.write("Symptoms:")

    symptoms=['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips', 'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness', 'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell ofurine', 'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain', 'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf', 'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze', 'prognosis']
    selected_symptoms = st.multiselect("Select symptoms:", symptoms)
    
    
    prediction = predict_disease(selected_symptoms, disease_predict)
# Display the predicted disease
    print("Predicted Disease:", prediction)





    # Display the selected symptoms with their appropriate indices
    st.write("Selected symptoms:")
    for symptom in selected_symptoms:
        index = symptoms.index(symptom)  # Get the index of the symptom in the original list
        st.write(f"{index}. {symptom}")
        
        
    button_clicked = st.button("Predict")
    if selected_symptoms and button_clicked:
            prediction = predict_disease(selected_symptoms, disease_predict)
            st.write("The predicted disease is: ", prediction)
    
        # # Output the prediction
        # if prediction == 1:
        #     st.write('Based on the provided symptoms, the predicted disease is: [Disease Name]')
        # else:
        #     st.write('Based on the provided symptoms, no specific disease could be predicted.')
    
    st.write("Choose your city:")   
    selected_city = st.multiselect("Choose your symptoms",["Ahmedabad", "Ajmer", "Aligarh", "Amritsar", "Ananthapur", "Bangalore", "Bareilly", "Belgaum", "Bellary", "Belthangady", "Bhiwandi", "Bhopal", "Bijnor", "Buldana", "Chandigarh", "Chellakere", "Chennai", "Cochin", "Coimbatore", "Cuddalore", "Cuttack", "Delhi", "Dibrugarh", "Eastgodavari", "Eluru", "Ernakulam", "Erode", "Faridabad", "Gandhinagar", "Ghaziabad", "Guntur", "Gurgaon", "Guwahati", "Himayatnagar", "Hosur", "Hyderabad", "Indore", "Jaipur", "Jammu", "Kanpur", "Karur", "Karwar", "Khurda", "Kolar", "Kolkata", "Koparkhairane", "Korba", "Kota", "Kozhicode", "Kundapur", "Kunnamkulam", "Kurnool", "Kurukshetra", "Kutch", "Lucknow", "Madurai", "Malapuram", "Manacaud", "Mandya", "Mangalore", "Margao", "Meerut", "Mehsana", "Miryalaguda", "Mohali", "Muktsar", "Mumbai", "Mysore", "Nagercoil", "Nagpur", "Nellore", "New Delhi", "Palakkad", "Panchkula", "Patan", "Patna", "Pondicherry", "Pune", "Raigad", "Raipur", "Rajamundry", "Rajkot", "Ramanagaram", "Ranchi", "Salem", "Sangli", "Secunderabad", "Shamshabad", "Sindhudurga", "Sriganganagar", "Srinagar", "Surat", "Tenali", "Thane", "Tirunelveli", "Trivandrum", "Tumkur", "Tuticorin", "Udupi", "Ulhasnagar", "Vaishali", "Varanasi", "Visakhapatnam", "Vizag", "Warangal", "Wardha"])
    clicked= st.button("Search Hospitals")
    
    if selected_city:
        if clicked:
            
            for city in selected_city:
                search_hospitals(city, hospital_suggest)
    
        
    
    
    
if __name__ == '__main__':
    main()
    
    



    





