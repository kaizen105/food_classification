import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Food AI 2.0: What & Where",
    page_icon="🧠",
    layout="wide",
)

# --- "GOOD CSS" (Styling) ---
def load_css():
    st.markdown(
        """
        <style>
        /* Main app container */
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
        }

        /* Title */
        h1 {
            color: #00A36C; /* A fresh green */
            text-align: center;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        
        /* Subheaders for results */
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
            height: 250px; /* Fixed height */
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
            display: block; /* Makes it take its own line */
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

# --- MODEL LOADING (Cached so it only runs once) ---

@st.cache_resource
def load_classifier():
    print("--- Loading 'nateraw/food' (101-Class) Model ---")
    classifier = pipeline("image-classification", model="nateraw/food")
    print("--- Classifier Loaded! ---")
    return classifier

@st.cache_resource
def load_detector():
    print("--- Loading 'yolov8n.pt' (Detector) Model ---")
    # This is the pre-trained model that knows 80 objects (incl. food)
    model = YOLO('yolov8n.pt') 
    print("--- Detector Loaded! ---")
    return model

# --- PREDICTION FUNCTIONS ---

def get_classification(classifier, image):
    """Gets the top classification guess."""
    results = classifier(image)
    top_guess = results[0]
    
    label = top_guess['label'].replace("_", " ").title()
    confidence = top_guess['score'] * 100
    return label, confidence

def get_detection_image(detector, image):
    """Runs detection and returns the image with boxes drawn on it."""
    # Convert PIL Image to numpy array
    img_np = np.array(image)
    
    # Run detection
    results = detector(img_np)
    
    # Get the first result and plot it (draws boxes)
    annotated_image_np = results[0].plot()
    
    # Convert the annotated numpy array (OpenCV BGR) back to PIL (RGB)
    annotated_image_pil = Image.fromarray(annotated_image_np[..., ::-1])
    return annotated_image_pil

# --- MAIN APP ---
def main():
    load_css()
    
    # Load models
    classifier = load_classifier()
    detector = load_detector()
    
    st.title("🧠 Food AI 2.0: What & Where")
    st.write("Upload an image to see what food it is (101 classes) AND where it is.")

    uploaded_file = st.file_uploader(
        "Upload a food image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Your Uploaded Image", use_column_width=True)
        with col2:
            st.write("") # Just for spacing

        if st.button("Analyze Food"):
            with st.spinner("🧠 AI is thinking... (running 2 models)"):
                
                # Run both models
                label, conf = get_classification(classifier, image)
                detected_image = get_detection_image(detector, image)
                
                # Display results in two new columns
                st.write("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.header("1. Classification (What it is)")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <p>My top guess is:</p>
                            <strong>{label}</strong>
                            <p class="confidence">({conf:.2f}% confidence)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_col2:
                    st.header("2. Detection (Where it is)")
                    st.image(detected_image, 
                             caption="Detected objects (from YOLOv8)", 
                             use_column_width=True)

if __name__ == "__main__":
    main()