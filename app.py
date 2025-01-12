import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from markup import doc_qa_tools_demo
from llm import get_suggestions_from_llm

# Load the saved SVM model
MODEL_PATH = "./saved_models/best_svm_model.joblib"
svm_model = joblib.load(MODEL_PATH)

def tab1():
    st.header(("Diabetes Predictor Tool"))
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie("https://lottie.host/28845468-6375-4bbb-a5b5-40f3742acfd1/puY00pXHQP.json")
       # st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(doc_qa_tools_demo(), unsafe_allow_html=True)

def predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dp_function, age):
    # Convert inputs to float and reshape for prediction
    input_features = [[
        float(pregnancies),
        float(glucose),
        float(blood_pressure),
        float(skin_thickness),
        float(insulin),
        float(bmi),
        float(dp_function),
        float(age)
    ]]
    
    # Perform prediction using the loaded model
    prediction = svm_model.predict(input_features)
    
    # Convert prediction to a readable result
    if prediction[0] == 1:
        return "Diabetic"
    else:
        return "Non-Diabetic"

def tab2():
    global patient_data
    st.header("Test for Diabetes")
    
    # Create a form to collect user input
    with st.form("diabetes_form"):
        pregnancies = st.number_input("Number of Pregnancies (integer):", min_value=0, max_value=25, step=1)
        glucose = st.number_input("Glucose Level (integer):", min_value=0, max_value=250, step=1)
        blood_pressure = st.number_input("Blood Pressure Level (integer):", min_value=0, max_value=150, step=1)
        skin_thickness = st.number_input("Skin Thickness (integer):", min_value=0, max_value=110, step=1)
        insulin = st.number_input("Insulin Level (integer):", min_value=0, max_value=1000, step=1)
        bmi = st.number_input("BMI Score (float):", min_value=0.0, max_value=80.0, step=0.1)
        dp_function = st.number_input("Diabetes Pedigree Function (float):", min_value=0.0, max_value=3.5, step=0.01)
        age = st.number_input("Age (integer):", min_value=0, max_value=100, step=1)

        # Submit button inside the form
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        with st.spinner("Testing for Diabetes..."):
            result = predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dp_function, age)
            st.success(f"Prediction: {result}")
            
            # Store patient's data and result in global variable
            patient_data = {
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "skin_thickness": skin_thickness,
                "insulin": insulin,
                "bmi": bmi,
                "dp_function": dp_function,
                "age": age,
                "prediction": result
            }

def tab3():
    global patient_data
    st.subheader("Suggestions by the LLM Model")
    
    if patient_data is None:
        st.warning("Please go to 'Test Diabetes' to make a prediction first.")
    else:
        # Display loading animation
        with st.spinner("Generating personalized health suggestions... This may take a few moments."):
            suggestions = get_suggestions_from_llm(patient_data)
        st.success(f"LLM Suggestions:\n{suggestions}")

def main():
    st.set_page_config(page_title="DocuBot", page_icon="📚", layout="wide")
    
    with st.sidebar:
        app_mode = option_menu(
            menu_title="Choose a page",  
            options=["Home", "Test Diabetes", "Suggestions for you"],
            icons=["house", "stethoscope", "lightbulb"],
            menu_icon="cast",  
            default_index=0,  
            orientation="vertical"  
        )
    
    # Page selection logic
    if app_mode == "Home":
        tab1()
    elif app_mode == "Test Diabetes":
        tab2()
    elif app_mode == "Suggestions for you":
        tab3()

if __name__ == "__main__":
    main()
