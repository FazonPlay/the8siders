import sys
import logging
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from gui.gui import CameraOCRGUI, QTextEditLogger
from image_capture.camera import Camera
from ocr_processing.text_recognition import OCRProcessor


class OCRWorker(QThread):
    resultReady = pyqtSignal(object, float, str)
    progressUpdate = pyqtSignal(int, str)
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, ocr_processor, image):
        super().__init__()
        self.ocr_processor = ocr_processor
        if image is not None:
            self.image = image.copy()
        else:
            self.image = None
            logging.error("Null image provided to OCRWorker")

        #program phases
    def run(self):
        try:
            #initial phase
            self.progressUpdate.emit(0, "Starting OCR engine...")
            self.progressChanged.emit(0)
            self.msleep(100)

            #orientation testing phase
            self.progressUpdate.emit(20, "Testing multiple image orientations...")
            self.progressChanged.emit(20)
            self.msleep(100)

            #preprocessing phase
            self.progressUpdate.emit(40, "Removing noise artifacts...")
            self.progressChanged.emit(40)
            self.msleep(100)

            #OCR phase
            self.progressUpdate.emit(60, "Applying specialized JD format detection...")
            self.progressChanged.emit(60)
            self.msleep(100)

            #safety check for image and processor
            if self.image is None:
                raise ValueError("No image data available")
            if self.ocr_processor is None:
                raise ValueError("OCR processor not available")

            text, confidence, method = self.ocr_processor.recognize_text(self.image)

            #post-processing phase
            self.progressUpdate.emit(80, "Compiling results from all methods...")
            self.progressChanged.emit(80)
            self.msleep(100)

            #results
            self.resultReady.emit(text, confidence, method)
            self.msleep(300)
            self.progressUpdate.emit(100, "Processing complete!")
            self.progressChanged.emit(100)

        except Exception as e:
            logging.error(f"OCR Worker error: {str(e)}")
            self.error.emit(str(e))
            self.errorOccurred.emit(str(e))
        finally:
            logging.info("OCR Worker finished")
            self.finished.emit()

def setup_logging(gui_log_widget=None):
    #prevent duplicate messages
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    #create log directory if not exist
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
        handlers=handlers
    )

def main():
    #catch unhandled exceptions
    def exception_hook(exctype, value, traceback):
        logging.error(f"Unhandled exception: {exctype.__name__}: {value}")
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        #create components
        camera = Camera()
        ocr_processor = OCRProcessor()
        gui = CameraOCRGUI(camera, ocr_processor)

        gui.ocr_worker_class = OCRWorker

        setup_logging(gui.log_text)
        logging.info("Starting application")
        logging.info("Initializing components")

        #initialize camera
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