import cv2
import time
import os


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Set camera properties for better image quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available

    def capture_image(self, save_path=None):
        if self.cap is None:
            self.initialize()

        # Allow camera to adjust to lighting
        time.sleep(1.0)

        # Capture multiple frames to let camera stabilize and choose best
        frames = []
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.1)

        if not frames:
            raise Exception("Failed to capture any images")

        # Use the last frame (most stabilized)
        frame = frames[-1]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, frame)

        return frame

    def capture_multiple(self, num_images=3, delay=0.5, save_dir="test_images"):
        """Capture multiple images with slight variations for better OCR results"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())

        images = []
        paths = []

        for i in range(num_images):
            path = os.path.join(save_dir, f"capture_{timestamp}_{i}.jpg")
            image = self.capture_image(save_path=path)
            images.append(image)
            paths.append(path)
            time.sleep(delay)

        return images, paths

    def release(self):
        if self.cap:
            self.cap.release()