import cv2
import time
import os


class Camera:
    def __init__(self, camera_index=0):
        # camera index 0 = laptop webcam, 1 = external webcam
        self.camera_index = camera_index
        self.cap = None
        self.width = 1920
        self.height = 1080

    @staticmethod
    def list_available_cameras(max_cameras=10):
        """List all available camera devices"""
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras

    def initialize(self):
        # Try DirectShow backend on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)  # Try default backend

        if not self.cap.isOpened():
            raise Exception(f"Could not open camera device at index {self.camera_index}")

        # Try to set resolution, with fallbacks
        resolutions = [(1920, 1080), (1280, 720), (800, 600), (640, 480)]

        for width, height in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Read a test frame to see if the resolution worked
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.width = width
                self.height = height
                break

        # Set autofocus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def capture_image(self, save_path=None):
        if self.cap is None or not self.cap.isOpened():
            self.initialize()

        # Allow camera to adjust to lighting
        for _ in range(5):  # Discard first few frames
            ret, _ = self.cap.read()
            time.sleep(0.1)

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, frame)

        return frame

    def capture_multiple(self, num_images=3, delay=0.5, save_dir="test_images"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())

        images = []
        paths = []

        for i in range(num_images):
            path = os.path.join(save_dir, f"capture_{timestamp}_{i}.jpg")
            image = self.capture_image(save_path=path)

            if image is not None:
                images.append(image)
                paths.append(path)

            time.sleep(delay)

        return images, paths

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()