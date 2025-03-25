# gui.py

import cv2
import os
import io
import sys
import logging
import time
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
        self.worker = None
        self.progress = None
        self.ocr_worker_class = None
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

    # In gui/gui.py
    # Keep only the more advanced version with error handling:

    def process_current_image(self):
        """Process the currently displayed image with worker thread"""
        if self.current_image is None:
            return

        # Disable processing button to prevent multiple processing
        self.process_button.setEnabled(False)
        self.results_text.append("Processing image...")

        try:
            # Create attractive progress dialog
            self.progress = QProgressDialog("Initializing OCR engine...", "Cancel", 0, 100, self)
            self.progress.setWindowTitle("OCR Processing")
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.setAutoClose(False)
            self.progress.setMinimumDuration(0)
            self.progress.setMinimumWidth(500)
            self.progress.setMinimumHeight(120)

            # Make the progress bar more visually interesting
            self.progress.setStyleSheet("""
                QProgressDialog {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 5px;
                }
                QProgressBar {
                    border: 1px solid #BDBDBD;
                    border-radius: 4px;
                    background-color: #e0e0e0;
                    text-align: center;
                    height: 25px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                    stop:0 #2196F3, stop:1 #03A9F4);
                    border-radius: 3px;
                }
                QLabel {
                    font-size: 13px;
                    color: #333;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)

            # Show progress dialog immediately
            self.progress.show()
            QApplication.processEvents()  # Force UI update

            # Check if worker class is available
            if self.ocr_worker_class is None:
                raise RuntimeError("OCR worker class not set")

            # Create worker thread with deep copy of image
            self.worker = self.ocr_worker_class(self.ocr_processor, self.current_image.copy())

            # Connect all signals before starting the thread
            self.worker.resultReady.connect(self.handle_ocr_result)
            self.worker.progressUpdate.connect(self.update_progress)
            self.worker.error.connect(self.handle_ocr_error)
            self.worker.finished.connect(self.cleanup_worker)

            # Add a cancellation option
            self.progress.canceled.connect(self.cancel_processing)

            # Start worker thread
            self.worker.start()

        except Exception as e:
            logging.error(f"Failed to start processing: {str(e)}")
            self.handle_ocr_error(f"Failed to start processing: {str(e)}")
            self.cleanup_worker()

    def handle_ocr_result(self, text, confidence, method):
        """Handle OCR results from worker thread"""
        # Update results
        result_text = f"Detected Text: {text}\n"
        result_text += f"Confidence: {confidence:.2f}%\n"
        result_text += f"Method: {method}\n"
        self.results_text.append(result_text)

        logging.info(f"Processed image: {text} ({confidence:.2f}%)")

    def update_progress(self, value, message):
        """Update progress dialog with detailed information and animation"""
        try:
            # Store a local reference to avoid race conditions
            progress_dialog = getattr(self, 'progress', None)

            # Skip if no progress dialog
            if progress_dialog is None:
                return

            try:
                # Set value and text
                progress_dialog.setValue(value)
                progress_dialog.setLabelText(message)

                # Log important progress points
                if value % 20 == 0 or value in [0, 100]:
                    logging.info(f"OCR Progress: {value}% - {message}")

                # Force UI update
                QApplication.processEvents()

            except RuntimeError:
                # Dialog was destroyed
                logging.warning("Progress dialog no longer available")

        except Exception as e:
            logging.error(f"Error updating progress: {str(e)}")

    def cleanup_worker(self):
        """Clean up after worker thread completes"""
        try:
            # Store local reference first
            local_worker = self.worker
            local_progress = getattr(self, 'progress', None)

            # Set class members to None immediately to prevent further updates
            self.worker = None
            self.progress = None

            # Wait for worker to finish if it's still running
            if local_worker and local_worker.isRunning():
                local_worker.wait(200)  # Wait up to 200ms

            # Close progress dialog last, after all updates should be done
            if local_progress is not None:
                try:
                    local_progress.close()
                except Exception as e:
                    logging.error(f"Error closing progress dialog: {str(e)}")

            # Re-enable processing button
            self.process_button.setEnabled(True)

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            self.process_button.setEnabled(True)

    def handle_ocr_error(self, error_message):
        """Handle errors from OCR worker"""
        try:
            self.results_text.append(f"ERROR: {error_message}")
            logging.error(f"OCR processing error: {error_message}")

            # Show error in results area with red highlight
            formatted_error = f"<span style='color: red; font-weight: bold;'>ERROR: {error_message}</span>"
            self.results_text.append(formatted_error)
        except Exception as e:
            logging.error(f"Error handling OCR error: {str(e)}")

    def cancel_processing(self):
        """Cancel current processing job"""
        try:
            logging.info("OCR processing canceled by user")

            # Store local references
            local_worker = self.worker
            local_progress = self.progress

            # Immediately set instance variables to None
            self.worker = None
            self.progress = None

            # Close progress dialog first to prevent updates
            if local_progress is not None:
                try:
                    local_progress.close()
                except Exception as e:
                    logging.error(f"Error closing progress dialog: {str(e)}")

            # Then terminate the worker thread
            if local_worker is not None and local_worker.isRunning():
                local_worker.running = False  # Signal to stop if thread checks this
                local_worker.wait(300)  # Wait for a clean exit

                # Force termination if still running
                if local_worker.isRunning():
                    local_worker.terminate()
                    local_worker.wait()

            # Re-enable the process button
            self.process_button.setEnabled(True)

        except Exception as e:
            logging.error(f"Error canceling processing: {str(e)}")
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
            # Only stop camera preview timer if it exists
            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()

            # Only stop process timer if it exists
            if hasattr(self, 'process_timer') and self.process_timer.isActive():
                self.process_timer.stop()

            # Release camera resources
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()

            logging.info("Application closed cleanly")
            event.accept()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            event.accept()

    def handle_ocr_result(self, text, confidence, method):
        """Handle OCR results from worker thread and show confirmation dialog"""
        # Update results in the main window
        result_text = f"Detected Text: {text}\n"
        result_text += f"Confidence: {confidence:.2f}%\n"
        result_text += f"Method: {method}\n"
        self.results_text.append(result_text)

        logging.info(f"Processed image: {text} ({confidence:.2f}%)")

        # Show confirmation dialog
        self.show_confirm_dialog(text, confidence)

    def show_confirm_dialog(self, text, confidence):
        """Show a dialog to confirm or modify the OCR result"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm OCR Result")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
                border-radius: 8px;
            }
            QLabel {
                font-size: 14px;
                padding: 10px;
            }
            QPushButton {
                min-width: 120px;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QLineEdit {
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            #resultLabel {
                font-size: 24px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout()

        # Add result label
        label = QLabel("Is this result correct?")
        layout.addWidget(label)

        # Show the result in large text
        result_label = QLabel(text)
        result_label.setObjectName("resultLabel")
        result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(result_label)

        # Add confidence
        conf_label = QLabel(f"Confidence: {confidence:.1f}%")
        conf_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(conf_label)

        # Buttons for confirmation
        button_layout = QHBoxLayout()

        # Green confirm button
        confirm_button = QPushButton("Correct")
        confirm_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(confirm_button)

        # Red edit button
        edit_button = QPushButton("Incorrect")
        edit_button.setStyleSheet("background-color: #F44336; color: white;")
        button_layout.addWidget(edit_button)

        layout.addLayout(button_layout)

        # Hidden text edit for corrections (initially hidden)
        text_edit = QLineEdit(text)
        text_edit.setVisible(False)
        layout.addWidget(text_edit)

        # Save button (initially hidden)
        save_button = QPushButton("Save Correction")
        save_button.setStyleSheet("background-color: #2196F3; color: white;")
        save_button.setVisible(False)
        layout.addWidget(save_button)

        dialog.setLayout(layout)

        # Connect events
        def on_confirm():
            logging.info(f"Result confirmed by user: {text}")
            self.save_final_result(text)
            dialog.accept()

        def on_edit():
            # Show editing interface
            result_label.setVisible(False)
            conf_label.setVisible(False)
            confirm_button.setVisible(False)
            edit_button.setVisible(False)

            # Show editing controls
            text_edit.setVisible(True)
            save_button.setVisible(True)

            # Update label
            label.setText("Please correct the result:")

        def on_save_correction():
            corrected_text = text_edit.text().strip()
            logging.info(f"Result corrected by user: {text} → {corrected_text}")
            self.save_final_result(corrected_text)
            dialog.accept()

        # Connect signals to slots
        confirm_button.clicked.connect(on_confirm)
        edit_button.clicked.connect(on_edit)
        save_button.clicked.connect(on_save_correction)

        # Execute dialog
        dialog.exec_()

    def save_final_result(self, final_text):
        """Save the final confirmed or corrected result to a separate file"""
        try:
            # Create results directory if it doesn't exist
            results_dir = "result_scan"
            os.makedirs(results_dir, exist_ok=True)

            # Create a filename based on the detected text and timestamp
            # Replace any invalid filename characters
            safe_text = ''.join(c for c in final_text if c.isalnum() or c in ' _-')
            safe_text = safe_text.strip().replace(' ', '_')

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Use detected text in filename if available, otherwise use timestamp only
            if safe_text:
                result_file = os.path.join(results_dir, f"{safe_text}_{timestamp}.txt")
            else:
                result_file = os.path.join(results_dir, f"scan_{timestamp}.txt")

            # Write the result to file
            with open(result_file, "w") as f:
                f.write(f"Final Result: {final_text}\n")
                f.write(f"Confidence: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Timestamp: {timestamp}\n")

            logging.info(f"Final result saved to {result_file}")

            # Show confirmation in results area
            self.results_text.append(f"\n✅ Result saved: {final_text}")
            self.results_text.append(f"📄 Saved to: {result_file}")

        except Exception as e:
            logging.error(f"Error saving final result: {str(e)}")
            self.results_text.append(f"❌ Error saving result: {str(e)}")