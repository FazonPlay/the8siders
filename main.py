
#  THIS VERSION OF THE MAIN IS IF YOU WANT TO IMPORT A FILE INSTEAD OF CAPTURING AN IMAGE


# import os
# import time
# import cv2
# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
# from ocr_processing.text_recognition import OCRProcessor
#
# def setup_directories():
#     """Create necessary directories for the project"""
#     dirs = ["results", "debug_images"]
#     for directory in dirs:
#         os.makedirs(directory, exist_ok=True)
#     return dirs
#
#
# def select_image_file():
#     """Open a file dialog to select an image"""
#     try:
#         root = tk.Tk()
#         root.withdraw()  # Hide the main window
#
#         # Make sure the dialog appears on top
#         root.attributes('-topmost', True)
#
#         # Print clear instruction
#         print("File selection dialog opening... (may appear behind other windows)")
#
#         file_path = filedialog.askopenfilename(
#             title="Select Image File",
#             filetypes=[
#                 ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
#                 ("All files", "*.*")
#             ]
#         )
#
#         root.destroy()
#         return file_path
#     except Exception as e:
#         print(f"Error opening file dialog: {e}")
#         # Fallback to manual input
#         print("Please enter the full path to your image file:")
#         return input("> ")
#
# def process_image(image_path, ocr_processor, display=True):
#     """Process a single image with OCR"""
#     if not image_path:
#         print("No image selected.")
#         return None
#
#     # Start timing
#     start_time = time.time()
#
#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not read image from {image_path}")
#         return None
#
#     # Recognize text
#     print(f"Processing image: {image_path}")
#     text, confidence, method = ocr_processor.recognize_text(image)
#
#     # Calculate total time
#     total_time = time.time() - start_time
#
#     # Save result
#     result_file = ocr_processor.save_result(text, confidence, method, image_path)
#
#     # Print results
#     print("\n===== OCR Results =====")
#     print(f"Text: {text}")
#     print(f"Confidence: {confidence:.2f}%")
#     print(f"Method: {method}")
#     print(f"Total processing time: {total_time:.3f}s")
#     print(f"Results saved to: {result_file}")
#
#     # Display the image with detected text if requested
#     if display:
#         display_img = image.copy()
#
#         # Resize if image is too large for display
#         max_height = 800
#         max_width = 1200
#         h, w = display_img.shape[:2]
#
#         if h > max_height or w > max_width:
#             scale = min(max_height / h, max_width / w)
#             display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
#             h, w = display_img.shape[:2]
#
#         # Create a black rectangle at the bottom
#         cv2.rectangle(display_img, (0, h - 100), (w, h), (0, 0, 0), -1)
#
#         # Add text with detection results
#         cv2.putText(display_img, f"Text: {text}", (10, h - 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(display_img, f"Confidence: {confidence:.2f}%", (10, h - 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(display_img, f"Time: {total_time:.3f}s", (10, h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         cv2.imshow("OCR Result", display_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     return text, confidence, method
#
#
# def main():
#     # Create necessary directories
#     setup_directories()
#
#     # Initialize OCR processor
#     ocr_processor = OCRProcessor()
#
#     # Prompt user to select an image file
#     print("Please select an image file to process...")
#     image_path = select_image_file()
#
#     if image_path:
#         # Process the selected image
#         process_image(image_path, ocr_processor, display=True)
#     else:
#         print("No image selected. Exiting...")
#
#
# if __name__ == "__main__":
#     main()
#
#
#


# THIS VERSION OF THE MAIN IS IF YOU WANT TO CAPTURE AN IMAGE INSTEAD OF IMPORTING A FILE


import os
import time
import cv2
import argparse
import numpy as np
from image_capture.camera import Camera
from ocr_processing.text_recognition import OCRProcessor


def setup_directories():
    """Create necessary directories for the project"""
    dirs = ["test_images", "results", "debug_images"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    return dirs


def process_image(image_path, ocr_processor, display=True):
    """Process a single image with OCR"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Recognize text
    print(f"Processing image: {image_path}")
    text, confidence, method = ocr_processor.recognize_text(image)

    # Save result
    result_file = ocr_processor.save_result(text, confidence, method, image_path)

    # Print results
    print("\n===== OCR Results =====")
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Method: {method}")
    print(f"Results saved to: {result_file}")

    # Display the image with detected text if requested
    if display:
        # Create a copy of the image for display
        display_img = image.copy()

        # Add text at the bottom of the image
        h, w = display_img.shape[:2]
        cv2.rectangle(display_img, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(display_img, f"Text: {text}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Confidence: {confidence:.2f}%", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the image
        cv2.imshow("OCR Result", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return text, confidence, method


def capture_and_process(camera, ocr_processor, num_captures=3, display=True):
    """Capture multiple images and process them with OCR"""
    # Capture multiple images
    print(f"Capturing {num_captures} images...")
    images, image_paths = camera.capture_multiple(num_images=num_captures)

    # Process each image
    results = []
    for img_path in image_paths:
        result = process_image(img_path, ocr_processor, display=False)
        if result:
            results.append(result + (img_path,))

    # Find the best result based on confidence
    if results:
        best_result = max(results, key=lambda x: x[1])
        text, confidence, method, best_image_path = best_result

        print("\n===== Best OCR Result =====")
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Method: {method}")
        print(f"Source: {best_image_path}")

        # Display the best result if requested
        if display:
            best_image = cv2.imread(best_image_path)

            # Add text at the bottom of the image
            h, w = best_image.shape[:2]
            cv2.rectangle(best_image, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(best_image, f"Text: {text}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(best_image, f"Confidence: {confidence:.2f}%", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the image
            cv2.imshow("Best OCR Result", best_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return text, confidence, method

    print("No valid OCR results found.")
    return None, 0, "no_results"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OCR System for Engraved Plates")
    parser.add_argument("--image", help="Path to image file to process (optional)")
    parser.add_argument("--no-display", action="store_true", help="Don't display results visually")
    parser.add_argument("--captures", type=int, default=3, help="Number of images to capture (default: 3)")
    args = parser.parse_args()

    # Create necessary directories
    setup_directories()

    # Initialize OCR processor
    ocr_processor = OCRProcessor()

    # Process a single image or capture from camera
    if args.image:
        # Process a specific image file
        process_image(args.image, ocr_processor, display=not args.no_display)
    else:
        # Capture and process from camera
        camera = Camera()
        try:
            capture_and_process(camera, ocr_processor, num_captures=args.captures,
                                display=not args.no_display)
        finally:
            camera.release()


if __name__ == "__main__":
    main()