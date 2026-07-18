from ultralytics import YOLO

print("Loading optimized OpenVINO model (this happens once)...")

# 1. --- THIS IS THE FIX ---
#    We MUST explicitly tell YOLO that this is a classification model.
model = YOLO('best_openvino_model/', task='classify') 

# 2. List of all the images you want to test
image_list = [
    'plate of fried chicken.jpg',
    'shrimp scampi pasta.jpg',
    'slice-of-chocholate-cake.jpg', # Make sure this filename is exact!
    'taco.jpeg',
    'tomato-soup-recipe.jpg'
]

print(f"--- Running predictions on {len(image_list)} images ---")

# 3. Loop through the list and predict each one
for image_file in image_list:
    try:
        # Run prediction
        results = model(image_file)

        # Get the results and print them
        result = results[0]             # Get the first result
        names = result.names            # Get the list of all food names
        top1_index = result.probs.top1  # Get the index (number) of the best guess
        top1_prob = result.probs.top1conf # Get the confidence of the best guess
        
        best_guess_name = names[top1_index] # Find the name for that index
        
        print(f"\nProcessing: {image_file}")
        print(f"==> I am {top1_prob.item()*100:.2f}% sure this is: {best_guess_name}")
    
    except Exception as e:
        # I've fixed the error message to be more accurate
        print(f"\n--- ERROR processing {image_file} ---")
        print(f"File was found, but the model failed during prediction.")
        print(f"Error details: {e}")


print("\n--- All predictions complete! ---")