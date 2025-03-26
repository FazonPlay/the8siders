#  THIS VERSION OF THE MAIN IS IF YOU WANT TO IMPORT A FILE INSTEAD OF CAPTURING AN IMAGE
import sys
import logging
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from gui.gui import CameraOCRGUI, QTextEditLogger
from image_capture.camera import Camera
from ocr_processing.text_recognition import OCRProcessor


class OCRWorker(QThread):
    """Worker thread to handle OCR processing"""
    resultReady = pyqtSignal(object, float, str)
    progressUpdate = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, ocr_processor, image):
        super().__init__()
        self.ocr_processor = ocr_processor
        self.image = image.copy()  # Create a deep copy
        self.running = True

    def run(self):
        try:
            # Initial phase
            self.progressUpdate.emit(0, "Starting OCR engine...")
            if not self.running: return
            self.msleep(100)

            # Orientation testing phase
            self.progressUpdate.emit(20, "Testing multiple image orientations...")
            if not self.running: return
            self.msleep(100)

            # Preprocessing phase
            self.progressUpdate.emit(40, "Removing noise artifacts...")
            if not self.running: return
            self.msleep(100)

            # OCR phase
            self.progressUpdate.emit(60, "Applying specialized JD format detection...")
            if not self.running: return
            self.msleep(100)

            # Actual processing happens here - check before starting actual processing
            if not self.running: return
            text, confidence, method = self.ocr_processor.recognize_text(self.image)

            # Post-processing phase
            self.progressUpdate.emit(80, "Compiling results from all methods...")
            if not self.running: return
            self.msleep(100)

            # Only emit results if not canceled
            if self.running:
                self.resultReady.emit(text, confidence, method)
                # Wait longer before finishing progress to allow dialog to show
                self.msleep(300)
                self.progressUpdate.emit(100, "Processing complete!")

        except Exception as e:
            logging.error(f"OCR Worker error: {str(e)}")
            if self.running:  # Only emit error if not canceled
                self.error.emit(str(e))
        finally:
            self.finished.emit()

    def update_progress(self, value, message):
        """Helper method to emit progress update and ensure UI processes it"""
        self.progressUpdate.emit(value, message)
        QApplication.processEvents()  # Process any pending events

def setup_logging(gui_log_widget=None):
    """Configure logging with both file and GUI output"""
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname('app.log')
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handlers = [
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]

    if gui_log_widget:
        handlers.append(QTextEditLogger(gui_log_widget))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


def main():
    """Main function to start the application"""

    # Enable exception hook to catch unhandled exceptions
    def exception_hook(exctype, value, traceback):
        logging.error(f"Unhandled exception: {exctype.__name__}: {value}")
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better progress dialogs

    try:
        # Create components
        camera = Camera()
        ocr_processor = OCRProcessor()
        gui = CameraOCRGUI(camera, ocr_processor)

        # Add OCR worker to GUI
        gui.ocr_worker_class = OCRWorker

        # Setup logging with GUI integration
        setup_logging(gui.log_text)
        logging.info("Starting application")
        logging.info("Initializing components")

        # Initialize camera safely
        # Camera initialization in main.py
        try:
            if hasattr(camera, 'initialize_camera'):
                camera.initialize_camera()
            elif hasattr(camera, 'initialize'):
                if not camera.initialize():
                    logging.warning("Camera initialization failed, will try to initialize on demand")
        except Exception as camera_error:
            logging.warning(f"Camera initialization error: {camera_error}. Will initialize on demand.")
        gui.show()
        logging.info("Application ready")

        return app.exec_()

    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        logging.error(error_msg)
        QMessageBox.critical(None, "Error", error_msg)
        return 1


if __name__ == '__main__':
    sys.exit(main())



# THIS VERSION OF THE MAIN IS IF YOU WANT TO CAPTURE AN IMAGE INSTEAD OF IMPORTING A FILE


# import os
# import time
# import cv2
# import argparse
# import numpy as np
# from image_capture.camera import Camera
# from ocr_processing.text_recognition import OCRProcessor
#
#
# def setup_directories():
#     """Create necessary directories for the project"""
#     dirs = ["test_images", "results", "debug_images"]
#     for directory in dirs:
#         os.makedirs(directory, exist_ok=True)
#     return dirs
#
#
# def process_image(image_path, ocr_processor, display=True):
#     """Process a single image with OCR"""
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
#     # Save result
#     result_file = ocr_processor.save_result(text, confidence, method, image_path)
#
#     # Print results
#     print("\n===== OCR Results =====")
#     print(f"Text: {text}")
#     print(f"Confidence: {confidence:.2f}%")
#     print(f"Method: {method}")
#     print(f"Results saved to: {result_file}")
#
#     # Display the image with detected text if requested
#     if display:
#         # Create a copy of the image for display
#         display_img = image.copy()
#
#         # Add text at the bottom of the image
#         h, w = display_img.shape[:2]
#         cv2.rectangle(display_img, (0, h - 80), (w, h), (0, 0, 0), -1)
#         cv2.putText(display_img, f"Text: {text}", (10, h - 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(display_img, f"Confidence: {confidence:.2f}%", (10, h - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         # Display the image
#         cv2.imshow("OCR Result", display_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     return text, confidence, method
#
#
# def capture_and_process(camera, ocr_processor, num_captures=3, display=True):
#     """Capture multiple images and process them with OCR"""
#     # Capture multiple images
#     print(f"Capturing {num_captures} images...")
#     images, image_paths = camera.capture_multiple(num_images=num_captures)
#
#     # Process each image
#     results = []
#     for img_path in image_paths:
#         result = process_image(img_path, ocr_processor, display=False)
#         if result:
#             results.append(result + (img_path,))
#
#     # Find the best result based on confidence
#     if results:
#         best_result = max(results, key=lambda x: x[1])
#         text, confidence, method, best_image_path = best_result
#
#         print("\n===== Best OCR Result =====")
#         print(f"Text: {text}")
#         print(f"Confidence: {confidence:.2f}%")
#         print(f"Method: {method}")
#         print(f"Source: {best_image_path}")
#
#         # Display the best result if requested
#         if display:
#             best_image = cv2.imread(best_image_path)
#
#             # Add text at the bottom of the image
#             h, w = best_image.shape[:2]
#             cv2.rectangle(best_image, (0, h - 80), (w, h), (0, 0, 0), -1)
#             cv2.putText(best_image, f"Text: {text}", (10, h - 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.putText(best_image, f"Confidence: {confidence:.2f}%", (10, h - 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#             # Display the image
#             cv2.imshow("Best OCR Result", best_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         return text, confidence, method
#
#     print("No valid OCR results found.")
#     return None, 0, "no_results"
#
#
# def main():
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="OCR System for Engraved Plates")
#     parser.add_argument("--image", help="Path to image file to process (optional)")
#     parser.add_argument("--no-display", action="store_true", help="Don't display results visually")
#     parser.add_argument("--captures", type=int, default=3, help="Number of images to capture (default: 3)")
#     args = parser.parse_args()
#
#     # Create necessary directories
#     setup_directories()
#
#     # Initialize OCR processor
#     ocr_processor = OCRProcessor()
#
#     # Process a single image or capture from camera
#     if args.image:
#         # Process a specific image file
#         process_image(args.image, ocr_processor, display=not args.no_display)
#     else:
#         # Capture and process from camera
#         camera = Camera()
#         try:
#             capture_and_process(camera, ocr_processor, num_captures=args.captures,
#                                 display=not args.no_display)
#         finally:
#             camera.release()
#
#
# if __name__ == "__main__":
#     main()