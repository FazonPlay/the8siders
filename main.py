import os
import time
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from ocr_processing.text_recognition import OCRProcessor

def setup_directories():
    """Create necessary directories for the project"""
    dirs = ["results", "debug_images"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    return dirs


def select_image_file():
    """Open a file dialog to select an image"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Make sure the dialog appears on top
        root.attributes('-topmost', True)

        # Print clear instruction
        print("File selection dialog opening... (may appear behind other windows)")

        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        root.destroy()
        return file_path
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        # Fallback to manual input
        print("Please enter the full path to your image file:")
        return input("> ")

def process_image(image_path, ocr_processor, display=True):
    """Process a single image with OCR"""
    if not image_path:
        print("No image selected.")
        return None

    # Start timing
    start_time = time.time()

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Recognize text
    print(f"Processing image: {image_path}")
    text, confidence, method = ocr_processor.recognize_text(image)

    # Calculate total time
    total_time = time.time() - start_time

    # Save result
    result_file = ocr_processor.save_result(text, confidence, method, image_path)

    # Print results
    print("\n===== OCR Results =====")
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Method: {method}")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Results saved to: {result_file}")

    # Display the image with detected text if requested
    if display:
        display_img = image.copy()

        # Resize if image is too large for display
        max_height = 800
        max_width = 1200
        h, w = display_img.shape[:2]

        if h > max_height or w > max_width:
            scale = min(max_height / h, max_width / w)
            display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
            h, w = display_img.shape[:2]

        # Create a black rectangle at the bottom
        cv2.rectangle(display_img, (0, h - 100), (w, h), (0, 0, 0), -1)

        # Add text with detection results
        cv2.putText(display_img, f"Text: {text}", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Confidence: {confidence:.2f}%", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Time: {total_time:.3f}s", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("OCR Result", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return text, confidence, method


def main():
    # Create necessary directories
    setup_directories()

    # Initialize OCR processor
    ocr_processor = OCRProcessor()

    # Prompt user to select an image file
    print("Please select an image file to process...")
    image_path = select_image_file()

    if image_path:
        # Process the selected image
        process_image(image_path, ocr_processor, display=True)
    else:
        print("No image selected. Exiting...")


if __name__ == "__main__":
    main()