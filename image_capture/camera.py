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












# import cv2
# import time
# import os
#
#
# class Camera:
#     def __init__(self, camera_index=0):
#         # camera index 0 = laptop webcam, 1 = external webcam
#         self.camera_index = camera_index
#         self.cap = None
#         self.width = 1920
#         self.height = 1080
#
#     @staticmethod
#     def list_available_cameras(max_cameras=10):
#         """List all available camera devices"""
#         available_cameras = []
#         for i in range(max_cameras):
#             cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#             if cap.isOpened():
#                 ret, _ = cap.read()
#                 if ret:
#                     available_cameras.append(i)
#                 cap.release()
#         return available_cameras
#
#     def initialize(self):
#         """Initialize the camera with better error handling"""
#         try:
#             # Release any existing capture
#             if self.cap is not None:
#                 self.cap.release()
#                 self.cap = None
#
#             # Try DirectShow backend first
#             self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
#
#             if not self.cap.isOpened():
#                 # Try default backend
#                 self.cap = cv2.VideoCapture(self.camera_index)
#
#             if not self.cap.isOpened():
#                 return False
#
#             # Set camera properties
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
#             self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
#
#             # Verify camera is working with a test capture
#             ret, frame = self.cap.read()
#             if not ret or frame is None:
#                 self.cap.release()
#                 self.cap = None
#                 return False
#
#             return True
#
#         except Exception as e:
#             if self.cap is not None:
#                 self.cap.release()
#                 self.cap = None
#             raise Exception(f"Camera initialization failed: {str(e)}")
#
#     def capture_image(self, save_path=None):
#         if self.cap is None or not self.cap.isOpened():
#             self.initialize()
#
#         # Clear buffer
#         for _ in range(5):
#             self.cap.grab()
#
#         frames = []
#         # Capture multiple frames to select the sharpest
#         for _ in range(10):
#             ret, frame = self.cap.read()
#             if ret and frame is not None:
#                 frames.append(frame)
#             time.sleep(0.1)
#
#         if not frames:
#             return None
#
#         # Select sharpest frame
#         best_frame = max(frames, key=lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
#
#         if save_path:
#             # Save with maximum quality
#             params = [cv2.IMWRITE_JPEG_QUALITY, 100]
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             cv2.imwrite(save_path, best_frame, params)
#
#         return best_frame
#
#     def capture_multiple(self, num_images=3, delay=0.5, save_dir="test_images"):
#         os.makedirs(save_dir, exist_ok=True)
#         timestamp = int(time.time())
#
#         images = []
#         paths = []
#
#         for i in range(num_images):
#             path = os.path.join(save_dir, f"capture_{timestamp}_{i}.jpg")
#             image = self.capture_image(save_path=path)
#
#             if image is not None:
#                 images.append(image)
#                 paths.append(path)
#
#             time.sleep(delay)
#
#         return images, paths
#
#     def release(self):
#         if self.cap and self.cap.isOpened():
#             self.cap.release()