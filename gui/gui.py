# gui.py
import cv2
import os
import io
import sys  # Add this import
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTextEdit, QSplitter, QMessageBox,
                            QFileDialog, QStackedWidget, QGridLayout, QScrollArea, QApplication)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)

# gui.py - Replace the ConsoleCapture import with this class
class ConsoleCapture(io.StringIO):
    def __init__(self, gui_log):
        super().__init__()
        self.gui_log = gui_log
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, text):
        self.original_stdout.write(text)  # Write to console
        if self.gui_log and text.strip():  # Only append non-empty lines
            self.gui_log.append(text.strip())

    def flush(self):
        self.original_stdout.flush()

class CameraOCRGUI(QMainWindow):
    def __init__(self, camera, ocr_processor):
        super().__init__()
        self.camera = camera
        self.ocr_processor = ocr_processor
        self.preview_timer = QTimer()
        self.process_timer = QTimer()
        self.is_processing = False
        self.camera_active = False
        self.current_image = None
        self.batch_images = []
        self.init_ui()
        self.setup_logging()

    def setup_logging(self):
        """Set up logging to QTextEdit"""
        log_handler = QTextEditLogger(self.log_text)
        logging.getLogger().addHandler(log_handler)

    def init_ui(self):
        """Initialize the user interface"""
        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowTitle('OCR System')

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create top panel
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        main_layout.addWidget(top_panel)

        # Mode selection buttons
        self.camera_button = QPushButton('Camera Mode')
        self.camera_button.clicked.connect(self.toggle_camera_mode)
        top_layout.addWidget(self.camera_button)

        self.file_button = QPushButton('Load Single Image')
        self.file_button.clicked.connect(self.load_image)
        top_layout.addWidget(self.file_button)

        self.batch_button = QPushButton('Capture Batch')
        self.batch_button.clicked.connect(self.start_batch_capture)
        self.batch_button.setEnabled(False)
        top_layout.addWidget(self.batch_button)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        # Main preview
        self.preview_label = QLabel()
        preview_width = int(screen.width() * 0.5)
        preview_height = int(screen.height() * 0.5)
        self.preview_label.setMinimumSize(preview_width, preview_height)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 2px solid gray;")
        left_layout.addWidget(self.preview_label)

        # Batch images grid
        self.batch_grid = QGridLayout()
        self.batch_labels = []
        batch_width = int(screen.width() * 0.2)
        batch_height = int(screen.height() * 0.2)
        for i in range(3):
            label = QLabel()
            label.setMinimumSize(batch_width, batch_height)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(Qt.AlignCenter)
            self.batch_labels.append(label)
            self.batch_grid.addWidget(label, 0, i)

        batch_widget = QWidget()
        batch_widget.setLayout(self.batch_grid)
        left_layout.addWidget(batch_widget)

        # Camera info
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        left_layout.addWidget(self.info_text)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)

        # Process buttons
        buttons_panel = QWidget()
        buttons_layout = QHBoxLayout(buttons_panel)

        self.process_button = QPushButton('Process Single')
        self.process_button.clicked.connect(self.process_current_image)
        self.process_button.setEnabled(False)
        buttons_layout.addWidget(self.process_button)

        self.process_batch_button = QPushButton('Process Batch')
        self.process_batch_button.clicked.connect(self.process_batch)
        self.process_batch_button.setEnabled(False)
        buttons_layout.addWidget(self.process_batch_button)

        right_layout.addWidget(buttons_panel)

        # Log display
        right_layout.addWidget(QLabel("Application Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        right_layout.addWidget(self.log_text)

        # OCR Results
        right_layout.addWidget(QLabel("OCR Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)

        # Set up timers
        self.preview_timer.timeout.connect(self.update_preview)
        self.process_timer.timeout.connect(self.process_frame)

        # Update preview label size based on screen
        preview_width = int(screen.width() * 0.5)  # 50% of screen width
        preview_height = int(screen.height() * 0.5)  # 50% of screen height
        self.preview_label.setMinimumSize(preview_width, preview_height)

        # Update batch image preview sizes
        batch_width = int(screen.width() * 0.2)  # 20% of screen width
        batch_height = int(screen.height() * 0.2)  # 20% of screen height
        for label in self.batch_labels:
            label.setMinimumSize(batch_width, batch_height)

    def toggle_camera_mode(self):
        """Toggle between camera mode and image mode"""
        if not self.camera_active:
            try:
                if self.camera.initialize():
                    self.camera_active = True
                    self.camera_button.setText('Stop Camera')
                    self.file_button.setEnabled(False)
                    self.batch_button.setEnabled(True)
                    self.process_button.setEnabled(False)
                    self.start_camera()
                else:
                    raise Exception("Failed to initialize camera")
            except Exception as e:
                logging.error(f"Camera error: {str(e)}")
                QMessageBox.critical(self, "Error", f"Camera error: {str(e)}")
        else:
            self.stop_camera()
            self.camera_button.setText('Camera Mode')
            self.file_button.setEnabled(True)
            self.batch_button.setEnabled(False)
            self.preview_label.clear()
            self.info_text.clear()
            self.camera_active = False

    def start_batch_capture(self):
        """Start capturing batch of 3 images"""
        self.batch_images = []
        self.results_text.clear()
        self.process_batch_button.setEnabled(False)

        try:
            images, _ = self.camera.capture_multiple(num_images=3, delay=1.0)
            self.batch_images = images

            # Display captured images in grid
            for i, image in enumerate(images):
                if image is not None:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.batch_labels[i].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.batch_labels[i].setPixmap(scaled_pixmap)

            self.process_batch_button.setEnabled(True)
            self.log_text.append("Captured 3 images successfully")

        except Exception as e:
            logging.error(f"Batch capture error: {str(e)}")
            self.log_text.append(f"Error during capture: {str(e)}")

    def process_batch(self):
        """Process all images in the batch"""
        if not self.batch_images:
            return

        try:
            self.results_text.clear()
            self.results_text.append("Processing batch of images...\n")

            for i, image in enumerate(self.batch_images):
                self.results_text.append(f"\nProcessing image {i + 1}/3:")
                text, confidence, method = self.ocr_processor.recognize_text(image)

                result_text = f"Image {i + 1} Results:\n"
                result_text += f"Detected Text: {text}\n"
                result_text += f"Confidence: {confidence:.2f}%\n"
                result_text += f"Method: {method}\n"
                result_text += "-" * 40 + "\n"

                self.results_text.append(result_text)

            self.results_text.append("\nBatch processing complete!")

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            self.results_text.append(f"Error during processing: {str(e)}")

    def load_image(self):
        """Load and display an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            try:
                # Load and store image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("Failed to load image")

                # Convert for display
                rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Update UI
                self.preview_label.setPixmap(scaled_pixmap)
                self.process_button.setEnabled(True)
                self.info_text.setText(f"Loaded image: {os.path.basename(file_path)}\n"
                                     f"Resolution: {w}x{h}")
                self.results_text.clear()

            except Exception as e:
                logging.error(f"File loading error: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def process_current_image(self):
        """Process the currently loaded image"""
        if not self.current_image is None:
            try:
                self.process_button.setEnabled(False)
                self.results_text.append("Processing image...")

                # Capture console output
                console_capture = ConsoleCapture(self.log_text)
                sys.stdout = console_capture
                sys.stderr = console_capture

                text, confidence, method = self.ocr_processor.recognize_text(self.current_image)

                # Restore original stdout/stderr
                sys.stdout = console_capture.original_stdout
                sys.stderr = console_capture.original_stderr

                result_text = f"Detected Text: {text}\n"
                result_text += f"Confidence: {confidence:.2f}%\n"
                result_text += f"Method: {method}\n"
                result_text += f"\nHistory:\n"

                self.add_history_to_results(result_text)
                logging.info(f"Processed image: {text} ({confidence:.2f}%)")

            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                self.results_text.append(f"Error: {str(e)}")
            finally:
                # Ensure stdout/stderr are restored
                if 'console_capture' in locals():
                    sys.stdout = console_capture.original_stdout
                    sys.stderr = console_capture.original_stderr
                self.process_button.setEnabled(True)

    def process_batch(self):
        """Process all images in the batch"""
        if not self.batch_images:
            return

        try:
            self.results_text.clear()
            self.results_text.append("Processing batch of images...\n")

            # Capture console output
            console_capture = ConsoleCapture(self.log_text)
            sys.stdout = console_capture
            sys.stderr = console_capture

            for i, image in enumerate(self.batch_images):
                self.results_text.append(f"\nProcessing image {i + 1}/3:")
                text, confidence, method = self.ocr_processor.recognize_text(image)

                result_text = f"Image {i + 1} Results:\n"
                result_text += f"Detected Text: {text}\n"
                result_text += f"Confidence: {confidence:.2f}%\n"
                result_text += f"Method: {method}\n"
                result_text += "-" * 40 + "\n"

                self.results_text.append(result_text)

            self.results_text.append("\nBatch processing complete!")

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            self.results_text.append(f"Error: {str(e)}")
        finally:
            # Restore original stdout/stderr
            if 'console_capture' in locals():
                sys.stdout = console_capture.original_stdout
                sys.stderr = console_capture.original_stderr

    def start_camera(self):
        """Initialize and start the camera preview"""
        try:
            if self.camera.cap is None or not self.camera.cap.isOpened():
                if not self.camera.initialize():
                    raise Exception("Failed to initialize camera")

            # Test camera capture
            ret, frame = self.camera.cap.read()
            if not ret or frame is None:
                raise Exception("Camera test capture failed")

            # Start preview timer (30 FPS)
            if self.preview_timer.isActive():
                self.preview_timer.stop()
            self.preview_timer.start(33)

            # Start processing timer (1 process every 3 seconds)
            if self.process_timer.isActive():
                self.process_timer.stop()
            self.process_timer.start(3000)

            logging.info("Camera started successfully")

        except Exception as e:
            logging.error(f"Camera initialization error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")
            self.close()

    def update_preview(self):
        """Update the camera preview"""
        if not self.camera.cap or not self.camera.cap.isOpened():
            self.preview_timer.stop()
            return

        try:
            ret, frame = self.camera.cap.read()
            if not ret or frame is None:
                self.preview_timer.stop()
                self.start_camera()
                return

            # Convert frame to RGB for Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Convert to QImage and scale
            image = QImage(rgb_frame.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(image).scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.preview_label.setPixmap(scaled_pixmap)
            self.update_camera_info(frame)

        except Exception as e:
            logging.error(f"Preview update error: {str(e)}")
            self.preview_timer.stop()

    def process_frame(self):
        """Process current frame with OCR"""
        if self.is_processing:
            return

        try:
            self.is_processing = True

            if self.camera.cap is None or not self.camera.cap.isOpened():
                raise Exception("Camera not available")

            # Update status
            self.results_text.append("Processing frame...")

            # Capture and process frame
            frame = self.camera.capture_image()
            if frame is not None:
                text, confidence, method = self.ocr_processor.recognize_text(frame)

                # Update results display
                result_text = f"Detected Text: {text}\n"
                result_text += f"Confidence: {confidence:.2f}%\n"
                result_text += f"Method: {method}\n"
                result_text += f"\nHistory:\n"

                # Add history from results directory
                self.add_history_to_results(result_text)

                logging.info(f"Processed frame: {text} ({confidence:.2f}%)")
            else:
                self.results_text.append("Failed to capture frame")

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.results_text.append(f"Error: {str(e)}")
        finally:
            self.is_processing = False

    def update_camera_info(self, frame):
        """Update camera information display"""
        info = f"Resolution: {frame.shape[1]}x{frame.shape[0]}\n"
        info += f"Camera Index: {self.camera.camera_index}\n"
        info += f"FPS: {self.camera.cap.get(cv2.CAP_PROP_FPS):.1f}"
        self.info_text.setText(info)

    def add_history_to_results(self, result_text):
        """Add history from results directory to display"""
        try:
            results_dir = "results"
            if os.path.exists(results_dir):
                files = sorted(os.listdir(results_dir), reverse=True)[:5]
                for file in files:
                    if file.startswith("result_") and file.endswith(".txt"):
                        try:
                            with open(os.path.join(results_dir, file), 'r') as f:
                                result_text += f"\n--- {file} ---\n"
                                result_text += f.read()
                        except Exception as e:
                            logging.error(f"Error reading history file {file}: {str(e)}")

            self.results_text.setText(result_text)
        except Exception as e:
            logging.error(f"Error updating history: {str(e)}")

    def stop_camera(self):
        """Stop camera preview and processing"""
        self.preview_timer.stop()
        self.process_timer.stop()
        if self.camera:
            self.camera.release()
        self.camera_active = False

    def closeEvent(self, event):
        """Clean up resources when closing"""
        try:
            self.stop_camera()
            logging.info("Application closed cleanly")
            event.accept()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            event.accept()