import os
import cv2 # OpenCV for image reading
import pytesseract # Tesseract wrapper
from PIL import Image # Pillow for image manipulation
import layoutparser as lp # For layout detection

# --- Configuration ---
INPUT_DIR = "screenshots"  # Folder with your images
OUTPUT_FILE = "smart_ocr_output_v2.txt" # Output file for all combined text (changed name to avoid overwrite)
# MODEL_CONFIG_PATH = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config" # A good model for general document layouts
# Alternative lighter model (might be faster, slightly less accurate on complex docs):
# MODEL_CONFIG_PATH = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
# MODEL_CONFIG_PATH = "lp://TableBank/faster_rcnn_R_101_FPN_3x/config" # If tables are primary

# Ensure the input directory exists
if not os.path.exists(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found. Please create it and add images.")
    exit()

# Initialize LayoutParser model
try:
    # --- START OF MODIFICATIONS FOR MANUALLY DOWNLOADED MODEL ---
    # --- START OF MODIFICATIONS FOR MANUALLY DOWNLOADED MODEL ---
    LOCAL_MODEL_WEIGHTS_FILENAME = "publaynet-tf_efficientdet_d0.pth.tar"
    LOCAL_MODEL_WEIGHTS_PATH = os.path.join("local_models", LOCAL_MODEL_WEIGHTS_FILENAME)

    # Check if the local model file exists
    if not os.path.exists(LOCAL_MODEL_WEIGHTS_PATH):
        print(f"ERROR: Local model weights file not found at: {LOCAL_MODEL_WEIGHTS_PATH}")
        print(f"Please download '{LOCAL_MODEL_WEIGHTS_FILENAME}' and place it in the 'local_models' directory.")
        print("You can usually find this model via a web search for 'publaynet tf_efficientdet_d0 pth.tar download' or from LayoutParser's model zoo documentation.")
        print("A common direct download link (may change over time) is: https://www.dropbox.com/s/ukbw5s673633hsw/publaynet-tf_efficientdet_d0.pth.tar?dl=1")
        exit()
    # ... rest of the model loading ...

    model = lp.models.EfficientDetLayoutModel(
        config_path='lp://PubLayNet/tf_efficientdet_d0/config',
        model_path=LOCAL_MODEL_WEIGHTS_PATH,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}, # Adjust based on your model's capabilities
        extra_config={"score_threshold": 0.5} # LOWERED threshold to detect more blocks
    )
    print(f"Successfully initialized EfficientDetLayoutModel using local weights from: {LOCAL_MODEL_WEIGHTS_PATH}")
    # --- END OF MODIFICATIONS FOR MANUALLY DOWNLOADED MODEL ---
except AttributeError as e:
    print(f"AttributeError during model initialization: {e}")
    print("This might mean layoutparser or its TensorFlow backend is not set up correctly, or the model class name is wrong.")
    print("Ensure 'tensorflow' is installed and you have a recent 'layoutparser' version.")
    exit()
except Exception as e_model: # Catch other potential model init errors
    print(f"An unexpected error occurred during model initialization: {e_model}")
    exit()


all_pages_text = []
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])

if not image_files:
    print(f"No image files found in '{INPUT_DIR}'.")
    exit()

print(f"Found {len(image_files)} images to process...")

for i, image_filename in enumerate(image_files):
    print(f"Processing image {i+1}/{len(image_files)}: {image_filename}")
    image_path = os.path.join(INPUT_DIR, image_filename)

    try:
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"  Warning: Could not read image {image_filename}. Skipping.")
            all_pages_text.append(f"[Could not read image: {image_filename}]\n\n----- END OF: {image_filename} -----\n\n")
            continue

        layout_result = model.detect(image_cv)
        text_blocks = layout_result # Use all detected blocks initially

        sorted_blocks = sorted(text_blocks, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        current_image_texts = [f"--- START OF: {image_filename} ---\n"]

        if not sorted_blocks:
            print(f"  No text blocks detected in {image_filename} by layout model. Attempting full page OCR.")
            try:
                # Preprocess full image slightly for better OCR
                gray_full = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                _, binary_full = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                pil_image_full = Image.fromarray(binary_full)
                full_page_text = pytesseract.image_to_string(pil_image_full, lang='eng', config='--psm 3').strip() # PSM 3 for auto page seg
                current_image_texts.append(full_page_text if full_page_text else "[Full page OCR - No text detected by Tesseract]")
            except Exception as e_ocr:
                print(f"    Error during full page OCR for {image_filename}: {e_ocr}")
                current_image_texts.append(f"[Error during full page OCR: {e_ocr}]")
        else:
            print(f"  Detected {len(sorted_blocks)} blocks in {image_filename}.")
            for block_num, block in enumerate(sorted_blocks):
                x1, y1, x2, y2 = map(int, block.coordinates)
                img_h, img_w = image_cv.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                if x1 >= x2 or y1 >= y2:
                    # print(f"    Skipping invalid block {block_num+1} (zero size after clamping).")
                    # current_image_texts.append(f"[Skipped invalid block {block_num+1} (zero size)]")
                    continue

                cropped_block_cv = image_cv[y1:y2, x1:x2]

                if cropped_block_cv is None or cropped_block_cv.size == 0:
                    # print(f"    Skipping empty block {block_num+1} after cropping.")
                    # current_image_texts.append(f"[Skipped empty block {block_num+1}]")
                    continue
                
                block_text_entry = f"[Block {block_num+1} - OCR Error or No Text]" # Default if something goes wrong

                try:
                    if cropped_block_cv.shape[0] > 0 and cropped_block_cv.shape[1] > 0:
                        # --- PREPROCESSING FOR EACH BLOCK ---
                        gray_crop = cv2.cvtColor(cropped_block_cv, cv2.COLOR_BGR2GRAY)
                        
                        h_crop, w_crop = gray_crop.shape
                        # Upscale if block is small
                        # Heuristic: if average dimension < 40px or height < 25px (more aggressive)
                        if (h_crop + w_crop) / 2 < 40 or h_crop < 25 : 
                             target_height = 50 # Aim for a minimum height for better OCR
                             scale = max(1.0, target_height / h_crop if h_crop > 0 else 1.0) # Avoid zero division
                             # Don't upscale excessively if width is already large relative to height
                             if w_crop * scale > img_w * 1.5 : # Cap excessive width scaling
                                 scale = (img_w * 1.5) / w_crop if w_crop > 0 else 1.0

                             if scale > 1.0: # Only resize if scaling up
                                #  print(f"    Upscaling block {block_num+1} by {scale:.2f}x")
                                 gray_crop = cv2.resize(gray_crop, (int(w_crop*scale), int(h_crop*scale)), interpolation=cv2.INTER_CUBIC)
                        
                        # Binarization (Otsu's method)
                        _, binary_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # DEBUG: Save preprocessed block to inspect what Tesseract sees
                        # debug_filename = f"debug_imgs/debug_{os.path.splitext(image_filename)[0]}_block{block_num+1}.png"
                        # os.makedirs("debug_imgs", exist_ok=True) # Create debug folder if it doesn't exist
                        # cv2.imwrite(debug_filename, binary_crop)

                        pil_block = Image.fromarray(binary_crop)
                        
                        # --- TESSERACT CONFIG ---
                        tesseract_config = '--oem 1 --psm 6' # LSTM engine, assume a single uniform block of text
                        
                        ocr_result = pytesseract.image_to_string(pil_block, lang='eng', config=tesseract_config).strip()
                        
                        if not ocr_result:
                            block_text_entry = f"[Block {block_num+1} - No text detected by Tesseract]"
                        else:
                            block_text_entry = ocr_result
                    else:
                        block_text_entry = f"[Block {block_num+1} (zero dimension)]"
                    
                    current_image_texts.append(block_text_entry)

                except cv2.error as e_cv: # Specifically catch OpenCV errors like the cvtColor one
                    print(f"    OpenCV Error OCRing block {block_num+1} in {image_filename}: {e_cv}")
                    current_image_texts.append(f"[OpenCV Error OCRing block {block_num+1}: {e_cv}]")
                except Exception as e_crop_ocr:
                    print(f"    Generic Error OCRing block {block_num+1} in {image_filename}: {e_crop_ocr}")
                    current_image_texts.append(f"[Error OCRing block {block_num+1}: {e_crop_ocr}]")

        all_pages_text.append("\n\n".join(current_image_texts))
        all_pages_text.append(f"\n\n----- END OF: {image_filename} -----\n\n")

    except Exception as e:
        print(f"  Major error processing image {image_filename}: {e}")
        all_pages_text.append(f"[Major error processing image: {image_filename} - {e}]\n\n----- END OF: {image_filename} -----\n\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(all_pages_text))

print(f"\nProcessing complete. Output saved to '{OUTPUT_FILE}'")