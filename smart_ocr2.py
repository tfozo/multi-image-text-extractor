import os
import cv2 # OpenCV for image reading
import pytesseract # Tesseract wrapper
from PIL import Image # Pillow for image manipulation
import layoutparser as lp # For layout detection

# --- Configuration ---
INPUT_DIR = "screenshots"  # Folder with your images
OUTPUT_FILE = "smart_ocr_output.txt" # Output file for all combined text
MODEL_CONFIG_PATH = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config" # A good model for general document layouts
# Alternative lighter model (might be faster, slightly less accurate on complex docs):
# MODEL_CONFIG_PATH = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
# MODEL_CONFIG_PATH = "lp://TableBank/faster_rcnn_R_101_FPN_3x/config" # If tables are primary

# Ensure the input directory exists
if not os.path.exists(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found. Please create it and add images.")
    exit()

# Initialize LayoutParser model (using TensorFlow-based EfficientDet is generally good on Windows)
# Using a simpler model that doesn't strictly require Detectron2 for Windows compatibility
# lp.EfficientDetLayoutModel uses TensorFlow. Ensure it's installed.
try:
    # If you installed layoutparser with TF support, EfficientDet is a good choice
    # You might need to install tensorflow if not already part of layoutparser[ocr] extras
    # pip install tensorflow

    # --- START OF MODIFICATIONS FOR MANUALLY DOWNLOADED MODEL ---
    LOCAL_MODEL_WEIGHTS_FILENAME = "publaynet-tf_efficientdet_d0.pth.tar" # The exact name of your downloaded file
    LOCAL_MODEL_WEIGHTS_PATH = os.path.join("local_models", LOCAL_MODEL_WEIGHTS_FILENAME)

    # Check if the local model file exists
    if not os.path.exists(LOCAL_MODEL_WEIGHTS_PATH):
        print(f"ERROR: Local model weights file not found at: {LOCAL_MODEL_WEIGHTS_PATH}")
        print(f"Please download '{LOCAL_MODEL_WEIGHTS_FILENAME}' and place it in the 'local_models' directory.")
        exit() # Or raise an exception

    model = lp.models.EfficientDetLayoutModel(
        config_path='lp://PubLayNet/tf_efficientdet_d0/config', # This tells LayoutParser the model ARCHITECTURE
        model_path=LOCAL_MODEL_WEIGHTS_PATH,                  # This tells LayoutParser to use YOUR downloaded WEIGHTS
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config={"score_threshold": 0.7}                 # Adjust confidence as needed
    )
    print(f"Successfully initialized EfficientDetLayoutModel using local weights from: {LOCAL_MODEL_WEIGHTS_PATH}")
    # --- END OF MODIFICATIONS FOR MANUALLY DOWNLOADED MODEL ---
except AttributeError:
    print("EfficientDetLayoutModel not found or TensorFlow backend issues.")
    print("Falling back to a basic Detectron2 model if available (may not work on Windows without setup).")
    # This is a fallback, Detectron2 is tricky on Windows.
    # If the above fails, it implies TensorFlow or the specific EfficientDet model path might be an issue.
    # For general text block detection, PubLayNet models are often used.
    model = lp.Detectron2LayoutModel( # This will likely require manual Detectron2 setup on Windows
        MODEL_CONFIG_PATH,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7], # Lower for more blocks, higher for fewer
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"} # Adjust based on model
    )
    print("Using Detectron2LayoutModel (fallback).")


all_pages_text = []

# Get a sorted list of image files
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])

if not image_files:
    print(f"No image files found in '{INPUT_DIR}'.")
    exit()

print(f"Found {len(image_files)} images to process...")

for i, image_filename in enumerate(image_files):
    print(f"Processing image {i+1}/{len(image_files)}: {image_filename}")
    image_path = os.path.join(INPUT_DIR, image_filename)

    try:
        # Read the image using OpenCV
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"  Warning: Could not read image {image_filename}. Skipping.")
            all_pages_text.append(f"[Could not read image: {image_filename}]\n\n----- END OF: {image_filename} -----\n\n")
            continue

        # Detect layout blocks
        # Convert BGR (OpenCV default) to RGB for some models if needed, but often works with BGR
        layout_result = model.detect(image_cv) # Pass the OpenCV image directly

        # Filter for text blocks (if your model distinguishes types)
        # For PubLayNet, type "Text" is common. If your model has other types, adjust.
        # If label_map was {0: "Text"}, then block.type might be 0 or "Text"
        # text_blocks = lp.Layout([b for b in layout_result if b.type == 'Text' or b.type == 0])
        # If you want all detected blocks regardless of type:
        text_blocks = layout_result # Use all detected blocks

        # Sort blocks by their y-coordinate, then x-coordinate (top-to-bottom, left-to-right)
        sorted_blocks = sorted(text_blocks, key=lambda block: (block.coordinates[1], block.coordinates[0]))

        current_image_texts = [f"--- START OF: {image_filename} ---\n"]

        if not sorted_blocks:
            print(f"  No text blocks detected in {image_filename}. Attempting full page OCR.")
            # Fallback to OCRing the whole image if no blocks are found
            try:
                pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                full_page_text = pytesseract.image_to_string(pil_image, lang='eng') # Specify language if needed
                current_image_texts.append(full_page_text.strip())
            except Exception as e_ocr:
                print(f"    Error during full page OCR for {image_filename}: {e}")
                current_image_texts.append(f"[Error during full page OCR: {e_ocr}]")
        else:
            print(f"  Detected {len(sorted_blocks)} blocks in {image_filename}.")
            for block_num, block in enumerate(sorted_blocks):
                # Get coordinates: x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, block.coordinates)

                # --- START OF ADDED VALIDATION ---
                # Ensure coordinates are valid and create a non-empty crop
                img_h, img_w = image_cv.shape[:2] # Get dimensions of the original image

                # Clamp coordinates to be within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                if x1 >= x2 or y1 >= y2: # Check for zero-width or zero-height
                    print(f"    Skipping invalid block {block_num+1} in {image_filename} (zero size after clamping: x1={x1},y1={y1},x2={x2},y2={y2}).")
                    current_image_texts.append(f"[Skipped invalid block {block_num+1} (zero size)]")
                    continue # Skip to the next block
                # --- END OF ADDED VALIDATION ---

                # Crop the image to this block using NumPy slicing on the OpenCV image
                cropped_block_cv = image_cv[y1:y2, x1:x2]

                # --- START OF ADDED CHECK FOR EMPTY CROP ---
                if cropped_block_cv is None or cropped_block_cv.size == 0:
                    print(f"    Skipping empty block {block_num+1} in {image_filename} after cropping.")
                    current_image_texts.append(f"[Skipped empty block {block_num+1}]")
                    continue # Skip to the next block
                # --- END OF ADDED CHECK FOR EMPTY CROP ---

                # Convert cropped OpenCV image (BGR) to PIL Image (RGB) for Pytesseract
                try:
                    # --- START OF ADDED CHECK (AGAIN, MORE ROBUST) ---
                    if cropped_block_cv.shape[0] > 0 and cropped_block_cv.shape[1] > 0: # Ensure width and height are positive
                        pil_block = Image.fromarray(cv2.cvtColor(cropped_block_cv, cv2.COLOR_BGR2RGB))
                        # OCR the cropped block
                        block_text = pytesseract.image_to_string(pil_block, lang='eng') # Specify language
                        current_image_texts.append(block_text.strip())
                    else:
                        print(f"    Skipping block {block_num+1} in {image_filename} due to zero dimension before cvtColor.")
                        current_image_texts.append(f"[Skipped block {block_num+1} (zero dimension)]")
                    # --- END OF ADDED CHECK ---
                except Exception as e_crop_ocr:
                    print(f"    Error OCRing block {block_num+1} in {image_filename}: {e_crop_ocr}")
                    current_image_texts.append(f"[Error OCRing block {block_num+1}: {e_crop_ocr}]")


        # Join texts from all blocks of this image with double newlines
        all_pages_text.append("\n\n".join(current_image_texts))
        # Add a clear separator for the end of this image's content
        all_pages_text.append(f"\n\n----- END OF: {image_filename} -----\n\n")

    except Exception as e:
        print(f"  Error processing image {image_filename}: {e}")
        all_pages_text.append(f"[Error processing image: {image_filename} - {e}]\n\n----- END OF: {image_filename} -----\n\n")

# Write all collected text to the output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(all_pages_text))

print(f"\nProcessing complete. Output saved to '{OUTPUT_FILE}'")