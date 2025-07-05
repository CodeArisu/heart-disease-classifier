import joblib as jl
import pandas as pd
import streamlit as st
from components.trajectory import processDataframe

classifier = 'heart_disease_classifier.pkl'
st_scaler = 'standard_scaler.pkl'

def init():
    try:
        heart_model = jl.load(filename=classifier)
        scaler = jl.load(filename=st_scaler)
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return

    st.title("Heart Disease Classification")
    st.write("Input details to predict the risk of heart disease.")
    
    # Collect user inputs

    col1, col2 = st.columns(2)

    with col1:
        st.radio(
            'Sex',
            ("Male", "Female"),
            key='sex',
            horizontal=True,
        )
        
        st.text_input("Patients BMI", key="bmi")
        
        st.text_input("Physical Health (score)", key="physical_health")
        
        st.text_input("Age range (eg. 10-20)", key="age_range")
        
        st.radio(
            'Smoker?',
            (True, False),
            key='is_smoking',
            horizontal=True,
        )
        
        st.radio(
            'Diabetic or (have been diagnose before?)',
            (True, False),
            key='is_diabetic',
            horizontal=True,
        )
    

    with col2:
        st.radio(
            'Does physical activities',
            (True, False),
            key='physical_activity',
            horizontal=True,
        )
        
        st.text_input("Sleep Time (hrs per day)", key="sleep_time")
        
        st.text_input("Mental Health (score)", key="mental_health")
        
        st.radio(
            'Current health status',
            ("poor", "fair", "good", "very good"),
            key='gen_health',
            horizontal=True,
        )
        
        st.radio(
            'Drinks alcohol?',
            (True, False),
            key='is_drinking',
            horizontal=True,
        )
        
        st.radio(
            'Has history of stoke?',
            (True, False),
            key='has_stroke',
            horizontal=True,
        )
    
    # Prepare test data (you should replace this with actual user inputs)
    test_data = {
        'bmi': st.session_state['bmi'],
        'smoking': st.session_state['is_smoking'],
        'alcoholdrinking': st.session_state['is_drinking'],
        'stroke': st.session_state['has_stroke'],
        'physicalhealth': st.session_state['physical_health'],
        'mentalhealth': st.session_state['mental_health'],
        'sex': st.session_state['sex'],
        'agecategory': st.session_state['age_range'],
        'diabetic': st.session_state['is_diabetic'],
        'physicalactivity': st.session_state['physical_activity'],
        'genhealth': st.session_state['gen_health'],
        'sleeptime': st.session_state['sleep_time'],
    }
    
    button('Predict', data=[{
        'model': heart_model,
        'scaler': scaler,
        }, test_data
    ])
    
def button(btnName, data):
    input_df = pd.DataFrame([data[1]])
    
    # Initialize state
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "proba" not in st.session_state:
        st.session_state.proba = None
    if "show_dialog" not in st.session_state: 
        st.session_state.show_dialog = False   
    if "input_text" not in st.session_state:   
        st.session_state.input_text = ''

    # Define button click handler
    def on_click():
        if not validate_inputs():
            return
        
        st.session_state.show_results = False
        st.session_state.show_dialog = True
        
        st.session_state.prediction, st.session_state.proba = loaded(
            model=data[0]['model'], 
            scaler=data[0]['scaler'], 
            inputs=input_df
        )
        
        st.session_state.show_results = True
        st.success("Prediction complete!")
        
    if st.button(btnName, on_click=on_click):
        if st.session_state.show_dialog:
            @st.dialog("Your Results")
            def dialog_box():
                if st.session_state.show_results and st.session_state.prediction is not None:
                    st.write(f"Prediction: {'Heart Disease' if st.session_state.prediction else 'No Heart Disease'}")
                    st.write(f"Probability: {st.session_state.proba[0]:.2%}")
                        
                if st.button("OK"):
                    st.session_state.show_dialog = False
                    st.rerun()
                    
            dialog_box()
        
def loaded(model, scaler, inputs):
    try:
        # Process the input data and make prediction
        prediction, proba = processDataframe(model, scaler, inputs)
        return prediction, proba
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None
    
def validate_inputs():
    required_text_fields = ['bmi', 'physical_health', 'age_range', 'sleep_time', 'mental_health']
    for field in required_text_fields:
        if not st.session_state.get(field, '').strip():
            st.error(f"Please fill in the {field.replace('_', ' ')} field")
            return False
    return True