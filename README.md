# AI Showdown: 3 Models 🧠

Welcome to the **Food Classification AI Showdown**! This Streamlit application compares the performance of three distinct machine learning models on food image classification. Upload an image of your favorite dish and see how each model interprets it.

## 🚀 The Models

1. **The "Pro" (Dish Expert)**: Uses `nateraw/food`, a model trained to recognize 101 specific food dishes (e.g., "Taco", "Sushi").
2. **Your "Avg" (Category Expert)**: A custom YOLO model (`best_openvino_model`) trained to classify images into 11 general food categories.
3. **The "Generalist" (Item Expert)**: Uses Google's Vision Transformer (`google/vit-base-patch16-224`), a general-purpose ImageNet model trained on 1000 common object classes.

## 📂 Project Structure

- `main.py`: The main Streamlit application script.
- `models/`: Contains the pre-trained weights for the custom YOLO model (`best.pt` and `best_openvino_model/`).
- `test_images/`: A collection of sample food images you can use to test the application.
- `scripts/`: Utility scripts used for model exporting and debugging.
- `requirements.txt`: Python dependencies required to run the application.
- `packages.txt`: System-level dependencies for Streamlit Cloud (e.g., `libgl1-mesa-glx` and `libglib2.0-0` for OpenCV).

## 🛠️ How to Run Locally

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/kaizen105/food_classification.git
   cd food_classification
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

## ☁️ Deployment

This app is designed to be easily deployed on **Streamlit Community Cloud**. The `packages.txt` file is included to ensure that system-level dependencies for `opencv-python-headless` are correctly installed in the cloud environment, preventing common `ImportError` issues.
