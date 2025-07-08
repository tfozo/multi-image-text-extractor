# Multi-Image Smart Text Extractor

This project uses Python with LayoutParser and Tesseract OCR to intelligently extract text from a collection of images. It identifies distinct text blocks within each image, OCRs them individually, and aggregates the results into a single output file, preserving the block structure.

## Features

*   Processes multiple images from a specified folder.
*   Utilizes LayoutParser with an EfficientDet model for robust text block detection.
*   Employs Tesseract OCR for text recognition within detected blocks.
*   Applies image preprocessing (grayscaling, conditional upscaling, binarization) to improve OCR accuracy on individual blocks.
*   Handles cases where no blocks are detected by attempting a full-page OCR.
*   Outputs all extracted text into a single, structured text file.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python:** Version 3.8 or higher. You can download it from [python.org](https://www.python.org/).
2.  **Git:** For version control and cloning the repository. You can download it from [git-scm.com](https://git-scm.com/).
3.  **Tesseract OCR Engine:**
    *   **Windows:**
        *   Download the installer from the [UB Mannheim Tesseract builds page](https://github.com/UB-Mannheim/tesseract/wiki) (e.g., `tesseract-ocr-w64-setup-v5.x.x.exe`).
        *   **Crucial:** During installation, make sure to check the box that says **"Add Tesseract to system PATH"** (or similar wording enabling system-wide access).
        *   You may also want to select additional language data packs during installation if you plan to OCR languages other than English.
    *   **Linux (Ubuntu/Debian-based):**
        ```bash
        sudo apt update
        sudo apt install tesseract-ocr tesseract-ocr-eng # For English
        # For other languages, e.g., French: sudo apt install tesseract-ocr-fra
        ```
    *   **macOS (using Homebrew):**
        ```bash
        brew install tesseract tesseract-lang
        ```
    *   **Verify Tesseract Installation:** Open a *new* terminal window after installation and type `tesseract --version`. You should see the Tesseract version information printed. If not, Tesseract was not added to your system PATH correctly.

## Setup Instructions

1.  **Clone the Repository:**
    Open your terminal and navigate to where you want to store the project. Then, clone the repository:
    ```bash
    git clone https://github.com/tfozo/multi-image-text-extractor.git
    cd multi-image-text-extractor
    ```

2.  **Manually Download the Layout Detection Model:**
    This project requires a pre-trained model for layout detection. You need to download it manually.
    *   **Model File Name:** `publaynet-tf_efficientdet_d0.pth.tar`
    *   **Download Link:** [Direct Download from Dropbox (approx. 80MB)](https://www.dropbox.com/s/ukbw5s673633hsw/publaynet-tf_efficientdet_d0.pth.tar?dl=1)
        *(Note: This link is provided for convenience and may change over time. If it's broken, search for "publaynet tf_efficientdet_d0 pth.tar download" or check LayoutParser documentation for official model sources.)*
    *   **Placement:**
        1.  In the root of the cloned project directory (`multi-image-text-extractor/`), create a new folder named `local_models`.
        2.  Place the downloaded `publaynet-tf_efficientdet_d0.pth.tar` file directly into this `local_models` folder. The script expects it at `local_models/publaynet-tf_efficientdet_d0.pth.tar`.

3.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv my_ocr_env 
    ```
    Activate the environment:
    *   **Windows (PowerShell or Command Prompt):**
        ```powershell
        .\my_ocr_env\Scripts\activate
        ```
    *   **macOS/Linux (Bash/Zsh):**
        ```bash
        source my_ocr_env/bin/activate
        ```
    Your terminal prompt should now be prefixed with `(my_ocr_env)`.

4.  **Install Python Dependencies:**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `layoutparser`, `tensorflow`, `opencv-python`, `Pillow`, `pytesseract`, and their dependencies as specified in the `requirements.txt` file.

5.  **Prepare Images:**
    *   Create a folder named `screenshots` in the root of the project directory (`multi-image-text-extractor/screenshots/`).
    *   Place all the images (e.g., `.png`, `.jpg`) you want to process into this `screenshots` folder.

## Running the Script

Ensure your virtual environment (`my_ocr_env`) is activated. Then, from the project root directory (`multi-image-text-extractor`), run the main script:

```bash
python smart_ocr2.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
