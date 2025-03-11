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