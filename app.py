
import joblib
best_algo = joblib.load('best_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_diabetes(glucose, bp, dpf, insulin, bmi):
    features = np.array([[glucose, bp, dpf, insulin, bmi]])
    features_scaled = scaler.transform(features)
    prediction = best_algo.predict(features_scaled)[0]
    probability = best_algo.predict_proba(features_scaled)[0]

    if prediction == 1:
        return f"DIABETIC ({probability[1]*100:.1f}% confidence)"
    else:
        return f"HEALTHY ({probability[0]*100:.1f}% confidence)"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
    """
    # Diabetes Prediction Test
    
    **Welcome! Click the button below and enter your medical details to begin your diabetes prediction.**
    """)
    
    start_btn = gr.Button("Test Begin", size="lg", variant="primary")
    
    with gr.Row(visible=False) as assessment_section:
        # LEFT COLUMN - Medical Inputs
        with gr.Column():
            gr.Markdown("### Medical Parameters")
            glucose = gr.Slider(70, 200, value=120, label="Glucose Level")
            bp = gr.Slider(60, 120, value=80, label="Blood Pressure")
            
            gr.Markdown("---")
            gr.Markdown("### Diabetes Pedigree Function")
            dpf = gr.Slider(0.0, 2.0, value=0.5, label="Genetic Risk Score")
            gr.Markdown("---")
            
            insulin = gr.Slider(0, 300, value=100, label="Insulin Level")
            bmi = gr.Slider(15, 40, value=25, label="BMI")
        
        # RIGHT COLUMN - Results
        with gr.Column():
            gr.Markdown("### Results")
            predict_btn = gr.Button("Analyze Diabetes Risk", size="lg", variant="secondary")
            result = gr.Textbox(label="Risk Assessment", interactive=False, lines=4)
    
    # Note section
    with gr.Column():
        gr.Markdown("---")
        gr.Markdown(
        """ 
        ### Important Note:
        - This test is for educational and testing purposes only
        - This tool provides basic assessment based on machine learning
        - It doesn't replace experts for accurate diagnosis
        - Please consult with healthcare provider for accurate diagnosis
        """)
    
    def show_assessment():
        return gr.update(visible=True)
    
    start_btn.click(
        fn=show_assessment,
        outputs=assessment_section
    )
    
    predict_btn.click(
        fn=predict_diabetes,
        inputs=[glucose, bp, dpf, insulin, bmi],
        outputs=result
    )

demo.launch(share=True)
