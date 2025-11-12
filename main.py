import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Food AI: Pro vs. Avg",
    page_icon="🤖",
    layout="wide",
)

# --- "GOOD CSS" (Styling) ---
def load_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
        }
        h1 {
            color: #00A36C; /* Fresh green */
            text-align: center;
            font-weight: bold;
        }
        h2 {
            color: #FAFAFA;
            border-bottom: 2px solid #00A36C;
            padding-bottom: 5px;
        }
        /* Result box for classification */
        .result-box {
            border: 2px solid #00A36C;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            font-size: 1.2em;
            background-color: #1c1f2b;
            height: 250px; /* Fixed height to align them */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .result-box p {
            font-size: 1.2em;
            margin: 0;
        }
        .result-box strong {
            font-size: 2.0em; /* Bigger font for the food name */
            color: #00A36C; /* Green */
            display: block;
            margin-top: 10px;
        }
        .confidence {
            font-size: 1.0em;
            color: #FAFAFA;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- MODEL 1: THE "PRO" (101-Class) LOADER ---
@st.cache_resource
def load_pro_classifier():
    print("--- Loading 'nateraw/food' (101-Class) Model ---")
    # This is the "genius" 0-minute model from Hugging Face
    classifier = pipeline("image-classification", model="nateraw/food")
    print("--- Pro Classifier Loaded! ---")
    return classifier

# --- MODEL 2: YOUR "AVG" (11-Class) LOADER ---
@st.cache_resource
def load_avg_classifier():
    print("--- Loading YOUR 'avg' (11-Class) Model ---")
    # This is your custom-trained OpenVINO model
    model = YOLO('best_openvino_model/', task='classify')
    print("--- Your Model Loaded! ---")
    return model

# --- PREDICTION FUNCTIONS ---

def get_pro_prediction(classifier, image):
    """Gets the "Specific Item" guess (e.g., "Taco")"""
    results = classifier(image)
    top_guess = results[0]
    label = top_guess['label'].replace("_", " ").title()
    confidence = top_guess['score'] * 100
    return label, confidence

def get_avg_prediction(classifier, image):
    """Gets the "General Category" guess (e.g., "Vegetable-Fruit")"""
    results = classifier(image)
    result = results[0]
    names = result.names
    top1_index = result.probs.top1
    top1_prob = result.probs.top1conf
    label = names[top1_index]
    confidence = top1_prob.item() * 100
    return label, confidence

# --- MAIN APP ---
def main():
    load_css()
    
    # Load both models
    pro_model = load_pro_classifier()
    avg_model = load_avg_classifier()
    
    st.title("🤖 Food AI: Pro vs. Avg")
    st.write("Which model is smarter? The 'Pro' (101 classes) or 'Your Avg' (11 classes)?")

    uploaded_file = st.file_uploader(
        "Upload a food image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Show the uploaded image
        st.image(image, caption="Your Uploaded Image", use_column_width=True, width=400)

        if st.button("Analyze Food"):
            with st.spinner("🤖🤖 Running two models..."):
                
                # Run both models on the same image
                pro_label, pro_conf = get_pro_prediction(pro_model, image)
                avg_label, avg_conf = get_avg_prediction(avg_model, image)
                
                st.write("---")
                col1, col2 = st.columns(2)
                
                # Column for the "Pro" 101-class model
                with col1:
                    st.header("1. 'Pro' 101-Class Model")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>It thinks the item is:</p>
                            <strong>{pro_label}</strong>
                            <p class="confidence">({pro_conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Column for "Your" 11-class model
                with col2:
                    st.header("2. 'Your Avg' 11-Class Model")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>It thinks the category is:</p>
                            <strong>{avg_label}</strong>
                            <p class="confidence">({avg_conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()