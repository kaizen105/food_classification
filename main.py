import streamlit as st
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Showdown: 3 Models",
    page_icon="🧠",
    layout="wide",
)

# --- "GOOD CSS" (Styling) ---
def load_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0E1117; }
        h1 { color: #00A36C; text-align: center; font-weight: bold; }
        h2 { color: #FAFAFA; border-bottom: 2px solid #00A36C; padding-bottom: 5px; }
        .result-box {
            border: 2px solid #00A36C; border-radius: 10px;
            padding: 25px; text-align: center;
            font-size: 1.2em; background-color: #1c1f2b;
            height: 250px; display: flex;
            flex-direction: column; justify-content: center;
        }
        .result-box p { font-size: 1.2em; margin: 0; }
        .result-box strong {
            font-size: 2.0em; color: #00A36C;
            display: block; margin-top: 10px;
        }
        .confidence { font-size: 1.0em; color: #FAFAFA; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- MODEL 1: THE "PRO" (101-Class Dish Expert) ---
@st.cache_resource
def load_pro_classifier():
    print("--- Loading 'nateraw/food' (101-Class) Model ---")
    classifier = pipeline("image-classification", model="nateraw/food")
    print("--- Pro Classifier Loaded! ---")
    return classifier

# --- MODEL 2: YOUR "AVG" (11-Class Category Expert) ---
@st.cache_resource
def load_avg_classifier():
    print("--- Loading YOUR 'avg' (11-Class) Model ---")
    model = YOLO('best_openvino_model/', task='classify')
    print("--- Your Model Loaded! ---")
    return model

# --- MODEL 3: THE "GENERALIST" (1000-Class Item Expert) ---
@st.cache_resource
def load_general_classifier():
    print("--- Loading 'google/vit-base-patch16-224' (ImageNet 1k) Model ---")
    # This is a standard Google Vision Transformer trained on ImageNet
    classifier = pipeline(
        "image-classification", 
        model="google/vit-base-patch16-224"
    )
    print("--- Generalist Classifier Loaded! ---")
    return classifier

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

def get_general_prediction(classifier, image):
    """Gets the "General Item" guess (e.g., "Apple")"""
    results = classifier(image)
    top_guess = results[0]
    # ImageNet labels can be complex, so we just take the first part
    label = top_guess['label'].split(",")[0].title()
    confidence = top_guess['score'] * 100
    return label, confidence

# --- MAIN APP ---
def main():
    load_css()
    
    # Load all three models
    pro_model = load_pro_classifier()
    avg_model = load_avg_classifier()
    general_model = load_general_classifier()
    
    st.title("🧠 AI Showdown: The 3-Model Test")
    st.write("Let's see what each model thinks your image is.")

    uploaded_file = st.file_uploader(
        "Upload a food image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Image", use_column_width=True, width=400)

        if st.button("Analyze Food"):
            with st.spinner("🤖🤖🤖 Running three models..."):
                
                # Run all models on the same image
                pro_label, pro_conf = get_pro_prediction(pro_model, image)
                avg_label, avg_conf = get_avg_prediction(avg_model, image)
                gen_label, gen_conf = get_general_prediction(general_model, image)
                
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.header("1. 'Pro' (101 Dishes)")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>Dish Expert says:</p>
                            <strong>{pro_label}</strong>
                            <p class="confidence">({pro_conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.header("2. 'Your Avg' (11 Categories)")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>Category Expert says:</p>
                            <strong>{avg_label}</strong>
                            <p class="confidence">({avg_conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    st.header("3. 'Generalist' (1k Items)")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>Item Expert says:</p>
                            <strong>{gen_label}</strong>
                            <p class="confidence">({gen_conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()