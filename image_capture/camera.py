# NEW CAMERA SYSTEM
# image_capture/camera.py
import cv2
import os
import time
import logging


class Camera:
    def __init__(self):
        self.cap = None
        self.current_camera_index = 0
        self.test_images_dir = "test_images"
        os.makedirs(self.test_images_dir, exist_ok=True)

    def initialize_camera(self):
        """Initialize camera connection"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.current_camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def switch_camera(self, camera_index):
        """Switch to a different camera"""
        # Release current camera if it exists
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Update camera index
        self.current_camera_index = camera_index

        # Initialize with new index
        return self.initialize_camera()

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def capture_single(self):
        """Capture a single image"""
        try:
            self.initialize_camera()
            # Warm up the camera
            for _ in range(5):
                self.cap.read()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture image")

            # Save the image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = os.path.join(self.test_images_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            return frame, image_path

        except Exception as e:
            logging.error(f"Camera capture error: {str(e)}")
            raise
        finally:
            self.release()

    def capture_multiple(self, num_images=3, delay=1.0):
        """Capture multiple images with delay"""
        images = []
        image_paths = []

        try:
            self.initialize_camera()
            # Warm up the camera
            for _ in range(5):
                self.cap.read()

            for i in range(num_images):
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Failed to capture image {i + 1}")

                # Save the image
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = os.path.join(self.test_images_dir, f"capture_{timestamp}_{i + 1}.jpg")
                cv2.imwrite(image_path, frame)

                images.append(frame)
                image_paths.append(image_path)
                time.sleep(delay)  # Wait between captures

            return images, image_paths

        except Exception as e:
            logging.error(f"Camera capture error: {str(e)}")
            raise
        finally:
            self.release()