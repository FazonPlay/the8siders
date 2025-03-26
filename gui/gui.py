# gui.py

import cv2
import os
import io
import sys
import logging
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QMessageBox,
                             QFileDialog, QGridLayout, QScrollArea, QApplication, QProgressDialog,
                             QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)


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


class OCRWorker(QThread):
    progressChanged = pyqtSignal(int)
    finished = pyqtSignal(str, float, str, float)
    errorOccurred = pyqtSignal(str)

    def __init__(self, processor, image):
        super().__init__()
        self.processor = processor
        self.image = image

    def run(self):
        try:
            start_time = time.time()
            for i in range(0, 80, 10):
                time.sleep(0.1)  # Simule le chargement progressif
                self.progressChanged.emit(i)
            text, confidence, method = self.processor.recognize_text(self.image)
            total_time = time.time() - start_time
            self.progressChanged.emit(100)
            self.finished.emit(text, confidence, method, total_time)
        except Exception as e:
            self.errorOccurred.emit(str(e))


class CameraOCRGUI(QMainWindow):
    def __init__(self, camera, ocr_processor):
        super().__init__()
        self.load_stylesheet("style.css")
        self.camera = camera
        self.ocr_processor = ocr_processor
        self.current_image = None
        self.worker = None
        self.progress = None
        self.ocr_worker_class = None
        self.batch_images = []
        self.processing_queue = []
        self.current_image_index = 0
        self.last_image_path = None
        self.processing_duration = 0
        self.is_user_cancelled = False

        # Window setup
        self.setWindowTitle("OCR Image Processor")

        # Get the available screen size
        desktop = QApplication.desktop()
        screen_rect = desktop.availableGeometry(self)

        # Set to maximum available screen size
        self.setGeometry(0, 0, screen_rect.width(), screen_rect.height())

        # Or maximize but keep window borders
        self.showMaximized()

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
        self.choose_camera_button = QPushButton("Choose Camera")
        self.choose_camera_button.setObjectName("choose_camera_button")
        self.choose_camera_button.setStyleSheet("background-color: #9C27B0; color: white;")

        # Add buttons to grid
        button_layout.addWidget(self.camera_button, 0, 0)
        button_layout.addWidget(self.batch_button, 0, 1)
        button_layout.addWidget(self.load_button, 1, 0)
        button_layout.addWidget(self.process_button, 1, 1)
        button_layout.addWidget(self.prev_button, 2, 0)
        button_layout.addWidget(self.next_button, 2, 1)
        button_layout.addWidget(self.choose_camera_button, 3, 0, 1, 2)  # Span both columns

        # Results and logging area
        results_label = QLabel("Results:")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        log_label = QLabel("Application Log:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        history_label = QLabel("History:")
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)

        # Add widgets to left panel
        left_layout.addWidget(button_panel)
        left_layout.addWidget(results_label)
        left_layout.addWidget(self.results_text)
        left_layout.addWidget(log_label)
        left_layout.addWidget(self.log_text)
        left_layout.addWidget(history_label)
        left_layout.addWidget(self.history_text)

        # Create right panel (image display)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)

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
        self.choose_camera_button.clicked.connect(self.show_camera_selector)

        # Initialize button states
        self.update_button_states()

    def load_stylesheet(self, filename):
            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_path, filename)
                with open(path, "r") as f:
                    self.setStyleSheet(f.read())
                logging.info(f"CSS chargÃ© depuis {path}")
            except Exception as e:
                logging.error(f"Erreur chargement CSS: {e}")

    def show_camera_selector(self):
        """Show a dialog to select which camera to use"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Camera")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Header
        label = QLabel("Select which camera to use:")
        layout.addWidget(label)

        # Camera selection dropdown
        camera_combo = QComboBox()

        # Check available cameras (0-5)
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                camera_name = f"Camera {i}"
                try:
                    # Try to get camera name (may not work on all systems)
                    ret, frame = cap.read()
                    if ret:
                        camera_name += f" ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})"
                except:
                    pass
                camera_combo.addItem(camera_name, i)
            cap.release()

        # Check if current camera is in list and select it
        current_index = self.camera.current_camera_index
        combo_index = camera_combo.findData(current_index)
        if combo_index >= 0:
            camera_combo.setCurrentIndex(combo_index)

        layout.addWidget(camera_combo)

        # No cameras found message
        if camera_combo.count() == 0:
            camera_combo.addItem("No cameras detected")
            camera_combo.setEnabled(False)

        # Button layout
        button_layout = QHBoxLayout()
        select_button = QPushButton("Select")
        select_button.setStyleSheet("background-color: #4CAF50; color: white;")
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("background-color: #F44336; color: white;")

        button_layout.addWidget(select_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Button event handlers
        def on_select():
            if camera_combo.count() > 0 and camera_combo.isEnabled():
                selected_index = camera_combo.currentData()
                if selected_index is not None and selected_index != self.camera.current_camera_index:
                    self.camera.current_camera_index = selected_index
                    # Release current camera to prepare for new one
                    self.camera.release()
                    logging.info(f"Selected camera changed to index {selected_index}")
                    self.results_text.append(f"Camera changed to: Camera {selected_index}")
            dialog.accept()

        def on_cancel():
            dialog.reject()

        # Connect signals
        select_button.clicked.connect(on_select)
        cancel_button.clicked.connect(on_cancel)

        # Execute dialog
        dialog.exec_()

    def capture_single_image(self):
        """Capture a single image from camera"""
        self.results_text.clear()
        try:
            image, image_path = self.camera.capture_single()
            if image is not None:
                self.current_image = image
                self.last_image_path = image_path
                self.batch_images = [image]
                self.current_image_index = 0
                self.display_current_image()
                self.results_text.append(f"Image captured successfully and saved to {image_path}")
                self.update_button_states()
        except Exception as e:
            self.results_text.append(f"Error capturing image: {str(e)}")
            logging.error(f"Error during image capture: {str(e)}")

    def capture_batch_images(self):
        """Capture multiple images in sequence"""
        self.results_text.clear()
        try:
            images, image_paths = self.camera.capture_multiple(num_images=3)
            if images:
                self.batch_images = images
                self.current_image = images[0]
                self.current_image_index = 0
                self.display_current_image()
                self.results_text.append(f"Captured {len(images)} images successfully")
                for i, path in enumerate(image_paths):
                    self.results_text.append(f"Image {i + 1}: {path}")
                self.update_button_states()
        except Exception as e:
            self.results_text.append(f"Error capturing batch: {str(e)}")
            logging.error(f"Error during batch capture: {str(e)}")

    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image = image
                    self.last_image_path = file_path
                    self.batch_images = [image]
                    self.current_image_index = 0
                    self.display_current_image()
                    self.results_text.append(f"Loaded image: {file_path}")
                    self.update_button_states()
                else:
                    raise ValueError(f"Failed to load image from {file_path}")
            except Exception as e:
                self.results_text.append(f"Error loading image: {str(e)}")
                logging.error(f"Error loading image: {str(e)}")

    def display_current_image(self):
        """Display the current image in the GUI"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            bytes_per_line = 3 * width
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Calculate scaling to fit in the display area while maintaining aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(scaled_pixmap)
            logging.info(f"Displayed image: {width}x{height}")

    def show_previous_image(self):
        """Show previous image in batch"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_image = self.batch_images[self.current_image_index]
            self.display_current_image()
            self.update_button_states()
            self.results_text.append(f"Showing image {self.current_image_index + 1} of {len(self.batch_images)}")

    def show_next_image(self):
        """Show next image in batch"""
        if self.current_image_index < len(self.batch_images) - 1:
            self.current_image_index += 1
            self.current_image = self.batch_images[self.current_image_index]
            self.display_current_image()
            self.update_button_states()
            self.results_text.append(f"Showing image {self.current_image_index + 1} of {len(self.batch_images)}")

    def update_button_states(self):
        """Update button states based on current application state"""
        has_image = self.current_image is not None
        self.process_button.setEnabled(has_image)
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.batch_images) - 1)

    def process_current_image(self):
        """Process the current image with OCR"""
        self.results_text.append("Processing image...")

        self.progress_dialog = QProgressDialog("Running OCR...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.cancel_processing)
        self.progress_dialog.show()

        self.worker = OCRWorker(self.ocr_processor, self.current_image)
        self.worker.progressChanged.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.ocr_finished)
        self.worker.errorOccurred.connect(self.ocr_failed)
        self.worker.start()

    def ocr_finished(self, text, confidence, method, duration):
        """Handle successful OCR completion"""
        self.progress_dialog.close()
        self.processing_duration = duration

        # If confidence is 90% or higher, automatically save without confirmation
        if confidence >= 90.0:
            self.results_text.append(f"High confidence result ({confidence:.1f}%) - Auto-saving")
            result_text = f"Detected Text: {text}\nConfidence: {confidence:.2f}%\nMethod: {method}\nProcessing time: {duration:.2f}s"
            self.results_text.append(result_text)

            # Save the results directly
            self.handle_ocr_result(text, confidence, method)
            self.save_final_result(text)
            self.save_to_history(text, confidence, method)
        else:
            # Show confirmation dialog for lower confidence results
            self.show_result_dialog(text, confidence, method)

        # Re-enable buttons
        self.update_button_states()

    def ocr_failed(self, error_message):
        """Handle OCR failure"""
        self.progress_dialog.close()
        self.results_text.append(f"Error during OCR: {error_message}")

    def cancel_processing(self):
        """Cancel the current OCR operation"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.results_text.append("OCR processing canceled")

    def reset_cancel_flag(self):
        """Reset the cancellation flag"""
        self.is_user_cancelled = False

    def show_result_dialog(self, text, confidence, method):
        """Show dialog with OCR results"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm OCR Result")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        info_label = QLabel(f"Confidence: {confidence:.1f}%\nMethod: {method}")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        result_label = QLabel(text)
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(result_label)

        edit_line = QLineEdit(text)
        edit_line.setVisible(False)
        layout.addWidget(edit_line)

        button_layout = QHBoxLayout()
        correct_btn = QPushButton("Correct")
        incorrect_btn = QPushButton("Incorrect")
        save_btn = QPushButton("Save Correction")
        save_btn.setVisible(False)

        button_layout.addWidget(correct_btn)
        button_layout.addWidget(incorrect_btn)
        layout.addLayout(button_layout)
        layout.addWidget(save_btn)

        def confirm():
            self.handle_ocr_result(text, confidence, method)
            self.save_to_history(text, confidence, method)
            dialog.accept()

        def edit():
            result_label.setVisible(False)
            correct_btn.setVisible(False)
            incorrect_btn.setVisible(False)
            edit_line.setVisible(True)
            save_btn.setVisible(True)

        def save():
            new_text = edit_line.text()
            self.handle_ocr_result(new_text, confidence, method)
            self.save_to_history(new_text, confidence, method)
            dialog.accept()

        correct_btn.clicked.connect(confirm)
        incorrect_btn.clicked.connect(edit)
        save_btn.clicked.connect(save)

        dialog.exec_()

    def handle_ocr_result(self, text, confidence, method):
        """Process OCR result and update UI"""
        result = f"Detected Text: {text}\nConfidence: {confidence:.2f}%\nMethod: {method}"
        self.results_text.append(result)
        logging.info(result)

    def save_to_history(self, final_text, confidence, method):
        """Save OCR result to history"""
        try:
            directory = "results"
            os.makedirs(directory, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_text = ''.join(c for c in final_text if c.isalnum() or c in ' _-').strip().replace(' ', '_')
            filename = f"{safe_text}_{timestamp}.txt" if safe_text else f"scan_{timestamp}.txt"
            path = os.path.join(directory, filename)

            with open(path, 'w') as f:
                f.write(f"Text: {final_text}\n")
                f.write(f"Confidence: {confidence:.2f}%\n")
                f.write(f"Method: {method}\n")
                f.write(f"Total processing time: {int(self.processing_duration)} seconds\n")

            self.history_text.append(f"Saved: {final_text} -> {filename}")
        except Exception as e:
            self.history_text.append(f" Error saving: {str(e)}")
            logging.error(f"Error saving result: {str(e)}")

    def save_final_result(self, final_text):
        """Save the final confirmed or corrected result to a separate file"""
        try:
            directory = "results"
            os.makedirs(directory, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Create a safe filename from text
            safe_text = ''.join(c for c in final_text if c.isalnum() or c in ' _-').strip()
            safe_text = safe_text.replace(' ', '_')[:30]  # Limit length

            if safe_text:
                filename = f"{safe_text}_{timestamp}.txt"
            else:
                filename = f"scan_{timestamp}.txt"

            result_file = os.path.join(directory, filename)

            with open(result_file, 'w') as f:
                f.write(f"Text: {final_text}\n")

                # Add additional information if available
                if hasattr(self, 'processing_duration') and self.processing_duration:
                    f.write(f"Processing time: {self.processing_duration:.2f}s\n")

                # Add image source if available
                if hasattr(self, 'last_image_path') and self.last_image_path:
                    f.write(f"Source image: {self.last_image_path}\n")

            logging.info(f"Final result saved to {result_file}")
            self.results_text.append(f" Saved to: {result_file}")

        except Exception as e:
            logging.error(f"Error saving final result: {str(e)}")
            self.results_text.append(f" Error saving result: {str(e)}")

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        try:
            # Stop any running workers
            if self.worker and self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()

            # Release camera if it exists
            if hasattr(self.camera, 'release'):
                self.camera.release()

            logging.info("Application closed")
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")

        event.accept()