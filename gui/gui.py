# gui.py
import cv2
import os
import io
import sys  # Add this import
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QSplitter, QMessageBox,
                             QFileDialog, QStackedWidget, QGridLayout, QScrollArea, QApplication, QProgressDialog)
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
        self.current_image = None
        self.batch_images = []
        self.processing_queue = []
        self.current_image_index = 0

        # Window setup
        self.setWindowTitle("OCR Image Processor")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                font-color: #000;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QLabel {
                font-size: 12px;
                color: #333;
            }
            QTextEdit {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                color: black;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left panel (controls and results)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        # Button panel
        button_panel = QWidget()
        button_layout = QGridLayout(button_panel)

        # Create and style buttons
        self.camera_button = QPushButton("Take Photo")
        self.batch_button = QPushButton("Take Batch (3)")
        self.load_button = QPushButton("Load Image")
        self.process_button = QPushButton("Process")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")

        # Add buttons to grid
        button_layout.addWidget(self.camera_button, 0, 0)
        button_layout.addWidget(self.batch_button, 0, 1)
        button_layout.addWidget(self.load_button, 1, 0)
        button_layout.addWidget(self.process_button, 1, 1)
        button_layout.addWidget(self.prev_button, 2, 0)
        button_layout.addWidget(self.next_button, 2, 1)

        # Results and logging area
        results_label = QLabel("Results:")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        log_label = QLabel("Application Log:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # Add widgets to left panel
        left_layout.addWidget(button_panel)
        left_layout.addWidget(results_label)
        left_layout.addWidget(self.results_text)
        left_layout.addWidget(log_label)
        left_layout.addWidget(self.log_text)

        # Create right panel (image display)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #333;
                border: 2px solid #BDBDBD;
                border-radius: 4px;
            }
        """)

        # Add image label to scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # Connect button signals
        self.camera_button.clicked.connect(self.capture_single_image)
        self.batch_button.clicked.connect(self.capture_batch_images)
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_current_image)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        # Initialize button states
        self.update_button_states()

    def capture_single_image(self):
        """Capture a single image from camera"""
        self.results_text.clear()
        try:
            image, _ = self.camera.capture_single()
            if image is not None:
                self.current_image = image
                self.batch_images = [image]
                self.current_image_index = 0
                self.display_current_image()
                self.results_text.append("Image captured successfully")
                self.update_button_states()
        except Exception as e:
            self.results_text.append(f"Error capturing image: {str(e)}")
            logging.error(f"Camera error: {str(e)}")

    def capture_batch_images(self):
        """Capture multiple images in sequence"""
        self.results_text.clear()
        try:
            images, _ = self.camera.capture_multiple(num_images=3)
            if images:
                self.batch_images = images
                self.current_image = images[0]
                self.current_image_index = 0
                self.display_current_image()
                self.results_text.append(f"Captured {len(images)} images")
                self.update_button_states()
        except Exception as e:
            self.results_text.append(f"Error capturing batch: {str(e)}")
            logging.error(f"Camera error: {str(e)}")

    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image = image
                    self.batch_images = [image]
                    self.current_image_index = 0
                    self.display_current_image()
                    self.results_text.append(f"Loaded image: {file_path}")
                    self.update_button_states()
                else:
                    raise ValueError("Failed to load image")
            except Exception as e:
                self.results_text.append(f"Error loading image: {str(e)}")
                logging.error(f"Image loading error: {str(e)}")

    def display_current_image(self):
        """Display the current image in the GUI"""
        if self.current_image is not None:
            # Convert image for display
            height, width = self.current_image.shape[:2]
            bytes_per_line = 3 * width
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(scaled_pixmap)

    def show_previous_image(self):
        """Show previous image in batch"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_image = self.batch_images[self.current_image_index]
            self.display_current_image()
            self.update_button_states()

    def show_next_image(self):
        """Show next image in batch"""
        if self.current_image_index < len(self.batch_images) - 1:
            self.current_image_index += 1
            self.current_image = self.batch_images[self.current_image_index]
            self.display_current_image()
            self.update_button_states()

    def update_button_states(self):
        """Update button states based on current context"""
        has_image = self.current_image is not None
        self.process_button.setEnabled(has_image)
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.batch_images) - 1)

    def show_processing_dialog(self):
        """Show a progress dialog during processing"""
        dialog = QProgressDialog("Processing image...", None, 0, 0, self)
        dialog.setWindowTitle("Processing")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        return dialog

    def process_current_image(self):
        """Process the currently displayed image"""
        if self.current_image is not None:
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
                self.results_text.append(result_text)

                logging.info(f"Processed image: {text} ({confidence:.2f}%)")

            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                self.results_text.append(f"Error: {str(e)}")
            finally:
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