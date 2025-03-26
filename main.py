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