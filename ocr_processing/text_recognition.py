import cv2
import numpy as np
import os
import time
from PIL import Image
import re
import difflib

# Try to import EasyOCR
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    raise ImportError("EasyOCR is required for this script. Install with: pip install easyocr")


class OCRProcessor:
    def __init__(self):
        self.last_result = None
        self.confidence = 0
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])
        else:
            raise RuntimeError("EasyOCR is not available. Please install it with: pip install easyocr")

    def preprocess_image_standard(self, image):
        """Standard preprocessing pipeline for OCR"""
        # Make a copy to avoid modifying the original
        img = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Save debug image
        cv2.imwrite(os.path.join(self.debug_dir, "1_standard_thresh.jpg"), thresh)

        return thresh

    def preprocess_image_enhanced(self, image):
        """Enhanced preprocessing pipeline for difficult text"""
        # Resize image (larger for better OCR)
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Save debug image
        cv2.imwrite(os.path.join(self.debug_dir, "2_enhanced_thresh.jpg"), opening)

        return opening

    def preprocess_image_inverse(self, image):
        """Preprocessing for white text on dark background"""
        # Resize image (larger for better OCR)
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary inverse thresholding
        _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Save debug image
        cv2.imwrite(os.path.join(self.debug_dir, "3_inverse_thresh.jpg"), binary_inv)

        return binary_inv

    def preprocess_image_adaptive(self, image):
        """Adaptive thresholding approach"""
        # Resize image
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding with larger block size
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 15, 8)

        # Save debug image
        cv2.imwrite(os.path.join(self.debug_dir, "4_adaptive_thresh.jpg"), adaptive)

        return adaptive

    def preprocess_for_printed_text(self, image):
        """Specialized preprocessing for clear printed text"""
        img = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)

        # Binary threshold with high value for clear text
        _, binary = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)

        # Save debug image
        cv2.imwrite(os.path.join(self.debug_dir, "5_printed_text.jpg"), binary)

        return binary

    def recognize_with_easyocr(self, image, preprocessing_method=None):
        """Recognize text using EasyOCR with optional preprocessing"""
        try:
            # Apply preprocessing if specified
            if preprocessing_method == "standard":
                processed_image = self.preprocess_image_standard(image)
            elif preprocessing_method == "enhanced":
                processed_image = self.preprocess_image_enhanced(image)
            elif preprocessing_method == "inverse":
                processed_image = self.preprocess_image_inverse(image)
            elif preprocessing_method == "adaptive":
                processed_image = self.preprocess_image_adaptive(image)
            elif preprocessing_method == "printed":
                processed_image = self.preprocess_for_printed_text(image)
            else:
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Run EasyOCR
            results = self.easyocr_reader.readtext(processed_image)

            if not results:
                return None, 0, f"easyocr_{preprocessing_method}_notext"

            texts = []
            confidences = []

            for (bbox, text, prob) in results:
                text = text.strip()
                if text and text.startswith("JD"):
                    texts.append(text)
                    confidences.append(prob * 100)

            if not texts:
                return None, 0, f"easyocr_{preprocessing_method}_no_jd"

            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Debug: log the result
            print(f"EasyOCR ({preprocessing_method}): '{full_text}' (Conf: {avg_confidence:.2f}%)")

            return full_text, avg_confidence, f"easyocr_{preprocessing_method}"

        except Exception as e:
            print(f"Error with EasyOCR ({preprocessing_method}): {e}")
            return None, 0, f"easyocr_{preprocessing_method}_error"

    def recognize_text(self, image):
        """Recognize text using EasyOCR with multiple preprocessing strategies"""
        results = []

        # Try different preprocessing methods
        preprocessing_methods = [
            None,  # No preprocessing
            "standard",
            "enhanced",
            "inverse",
            "adaptive",
            "printed"
        ]

        for method in preprocessing_methods:
            result = self.recognize_with_easyocr(image, method)
            if result[0]:  # If text was detected
                results.append(result)

        if not results:
            return "No text detected", 0, "no_valid_results"

        # If only one valid result, return it
        if len(results) == 1:
            return results[0]

        # Strategy 1: Select result with highest confidence
        best_by_confidence = max(results, key=lambda x: x[1])

        # Strategy 2: Check if multiple methods detected similar text
        texts = [r[0] for r in results]
        text_matches = {}

        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i < j:
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                    if similarity > 0.7:  # Texts are similar
                        # Choose the one with higher confidence
                        if results[i][1] > results[j][1]:
                            text_matches[text1] = text_matches.get(text1, 0) + 1
                        else:
                            text_matches[text2] = text_matches.get(text2, 0) + 1

        # If we found similar texts across preprocessing methods
        if text_matches:
            most_common_text = max(text_matches.items(), key=lambda x: x[1])[0]
            # Find the result with this text
            for result in results:
                if result[0] == most_common_text:
                    return result

        # Default to highest confidence
        return best_by_confidence

    def save_result(self, text, confidence, method, image_path=None, output_dir="results"):
        """Save OCR result to a file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = os.path.basename(image_path).split('.')[0] if image_path else str(int(time.time()))

        result_file = os.path.join(output_dir, f"result_{timestamp}.txt")
        with open(result_file, 'w') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Confidence: {confidence:.2f}%\n")
            f.write(f"Method: {method}\n")
            if image_path:
                f.write(f"Source: {image_path}\n")

        return result_file