from ultralytics import YOLO

print("Loading your trained model...")
# 1. Load your downloaded classification model
model = YOLO('best.pt') 

print("Exporting to OpenVINO format (for your Iris Xe GPU)...")
# 2. Export it!
model.export(format='openvino')

print("\n--- Export complete! ---")
print("A new folder named 'best_openvino_model' was created.")