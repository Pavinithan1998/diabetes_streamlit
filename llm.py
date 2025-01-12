from transformers import pipeline

# Load HuggingFace model for LLaMA or GPT-style inference
def get_suggestions_from_llm(patient_data):
    # Prepare prompt with patient's data
    prompt = (
        f"Patient details:\n"
        f" - Pregnancies: {patient_data['pregnancies']}\n"
        f" - Glucose Level: {patient_data['glucose']}\n"
        f" - Blood Pressure: {patient_data['blood_pressure']}\n"
        f" - Skin Thickness: {patient_data['skin_thickness']}\n"
        f" - Insulin Level: {patient_data['insulin']}\n"
        f" - BMI: {patient_data['bmi']}\n"
        f" - Diabetes Pedigree Function: {patient_data['dp_function']}\n"
        f" - Age: {patient_data['age']}\n"
        f"Prediction Result: {patient_data['prediction']}\n\n"
        f"Provide personalized health advice for this patient."
    )

    # Load the LLaMA model
    model_name = "huggingface/llama-7b"  # Change to the model you want
    llm_pipeline = pipeline("text-generation", model=model_name, max_new_tokens=200)
    
    # Generate response
    response = llm_pipeline(prompt)[0]["generated_text"]
    
    return response
