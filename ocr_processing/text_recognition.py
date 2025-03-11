import cv2
import numpy as np
import pytesseract
import os
import time
from PIL import Image
import re
import difflib
pytesseract.pytesseract.tesseract_cmd = r'C:\Installed Apps\tesseract.exe'  # Update this path to match your installation
# Try to import optional OCR engines
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class OCRProcessor:


    def __init__(self):
        self.last_result = None
        self.confidence = 0
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize optional OCR engines
        if EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])

        if PADDLEOCR_AVAILABLE:
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

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

    def recognize_with_tesseract(self, image, configs=None):
        """Recognize text using Tesseract with various configurations"""
        if configs is None:
            configs = [
                '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                '--oem 3 --psm 13',  # Raw line
                '--oem 1 --psm 10'  # Treat as single character, legacy engine
            ]

        results = []

        # Try different preprocessing methods
        preprocessed_images = [
            self.preprocess_image_standard(image),
            self.preprocess_image_enhanced(image),
            self.preprocess_image_inverse(image),
            self.preprocess_image_adaptive(image),
            self.preprocess_for_printed_text(image)  # Add this line

        ]

        for idx, proc_img in enumerate(preprocessed_images):
            for config_idx, config in enumerate(configs):
                try:
                    data = pytesseract.image_to_data(proc_img, config=config,
                                                     output_type=pytesseract.Output.DICT)

                    # Extract text and confidence
                    texts = []
                    confidences = []

                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 0 and data['text'][i].strip():
                            texts.append(data['text'][i])
                            confidences.append(int(data['conf'][i]))

                    # Combine text and calculate average confidence
                    result_text = ' '.join([t for t in texts if t.strip()])
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    if result_text.strip():
                        results.append({
                            'text': result_text,
                            'confidence': avg_confidence,
                            'method': f'tesseract_prep{idx}_config{config_idx}'
                        })

                        # Debug: log the result
                        print(
                            f"Tesseract (Prep {idx}, Config {config_idx}): '{result_text}' (Conf: {avg_confidence:.2f}%)")

                except Exception as e:
                    print(f"Error with Tesseract config {config}: {e}")

        if not results:
            return None, 0, "tesseract_failed"

        # Return the result with highest confidence
        best_result = max(results, key=lambda x: x['confidence'])
        return best_result['text'], best_result['confidence'], best_result['method']

    def recognize_with_easyocr(self, image):
        """Recognize text using EasyOCR"""
        if not EASYOCR_AVAILABLE:
            return None, 0, "easyocr_not_available"

        try:
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Run EasyOCR
            results = self.easyocr_reader.readtext(gray)

            if not results:
                return None, 0, "easyocr_no_text"

            texts = []
            confidences = []

            for (bbox, text, prob) in results:
                if text.strip():
                    texts.append(text)
                    confidences.append(prob * 100)  # Convert to percentage

            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Debug: log the result
            print(f"EasyOCR: '{full_text}' (Conf: {avg_confidence:.2f}%)")

            return full_text, avg_confidence, "easyocr"

        except Exception as e:
            print(f"Error with EasyOCR: {e}")
            return None, 0, "easyocr_error"



    def recognize_with_paddleocr(self, image):
        """Recognize text using PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            return None, 0, "paddleocr_not_available"

        try:
            # Run PaddleOCR
            result = self.paddleocr_reader.ocr(image, cls=True)

            if not result or not result[0]:
                return None, 0, "paddleocr_no_text"

            texts = []
            confidences = []

            for line in result[0]:
                text = line[1][0]  # Extract text
                prob = line[1][1] * 100  # Extract confidence and convert to percentage

                if text.strip():
                    texts.append(text)
                    confidences.append(prob)

            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Debug: log the result
            print(f"PaddleOCR: '{full_text}' (Conf: {avg_confidence:.2f}%)")

            return full_text, avg_confidence, "paddleocr"

        except Exception as e:
            print(f"Error with PaddleOCR: {e}")
            return None, 0, "paddleocr_error"

    def get_best_result(self, results):
        """Determine the best result from multiple OCR engines"""
        # Filter out None results
        valid_results = [r for r in results if r[0] is not None]

        if not valid_results:
            return "No text detected", 0, "no_valid_results"

        # If only one valid result, return it
        if len(valid_results) == 1:
            return valid_results[0]

        # Strategy 1: Select result with highest confidence
        best_by_confidence = max(valid_results, key=lambda x: x[1])

        # Strategy 2: Check if multiple engines detected the same or similar text
        texts = [r[0] for r in valid_results]
        text_matches = {}

        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i < j:
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                    if similarity > 0.7:  # Texts are similar
                        # Choose the one with higher confidence
                        if valid_results[i][1] > valid_results[j][1]:
                            text_matches[text1] = text_matches.get(text1, 0) + 1
                        else:
                            text_matches[text2] = text_matches.get(text2, 0) + 1

        # If we found similar texts across engines
        if text_matches:
            most_common_text = max(text_matches.items(), key=lambda x: x[1])[0]
            # Find the result with this text
            for result in valid_results:
                if result[0] == most_common_text:
                    return result

        # Default to highest confidence
        return best_by_confidence

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

    def recognize_text(self, image):
        """Recognize text using multiple OCR engines and strategies"""
        results = []

        # Use Tesseract
        tesseract_result = self.recognize_with_tesseract(image)
        if tesseract_result[0]:
            results.append(tesseract_result)

        # Use EasyOCR if available
        if EASYOCR_AVAILABLE:
            easyocr_result = self.recognize_with_easyocr(image)
            if easyocr_result[0]:
                results.append(easyocr_result)

        # Use PaddleOCR if available
        if PADDLEOCR_AVAILABLE:
            paddleocr_result = self.recognize_with_paddleocr(image)
            if paddleocr_result[0]:
                results.append(paddleocr_result)

        # Get the best result
        text, confidence, method = self.get_best_result(results)

        self.last_result = text
        self.confidence = confidence

        return text, confidence, method

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