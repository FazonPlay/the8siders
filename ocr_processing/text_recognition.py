import cv2
import numpy as np
import os
import time
import re
from PIL import Image
import difflib

# Try to import OCR engines
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR is not available. Install with: pip install easyocr")

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r'C:\Installed Apps\tesseract.exe'

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract is not available. Install with: pip install pytesseract")

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("PaddleOCR is not available. Install with: pip install paddleocr")


class OCRProcessor:
    def __init__(self):
        self.last_result = None
        self.confidence = 0
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize OCR engines
        if EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])

        if PADDLE_AVAILABLE:
            self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        # Check if at least one OCR engine is available
        if not (EASYOCR_AVAILABLE or TESSERACT_AVAILABLE or PADDLE_AVAILABLE):
            raise RuntimeError(
                "No OCR engine available. Please install at least one: easyocr, pytesseract, or paddleocr")

    def preprocess_image_standard(self, image):
        """Standard preprocessing pipeline for OCR"""
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(os.path.join(self.debug_dir, "1_standard_thresh.jpg"), thresh)
        return thresh

    def preprocess_image_enhanced(self, image):
        """Enhanced preprocessing pipeline for difficult text"""
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(self.debug_dir, "2_enhanced_thresh.jpg"), opening)
        return opening

    def preprocess_image_inverse(self, image):
        """Preprocessing for white text on dark background"""
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(os.path.join(self.debug_dir, "3_inverse_thresh.jpg"), binary_inv)
        return binary_inv

    def preprocess_image_adaptive(self, image):
        """Adaptive thresholding approach"""
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 15, 8)
        cv2.imwrite(os.path.join(self.debug_dir, "4_adaptive_thresh.jpg"), adaptive)
        return adaptive

    def preprocess_for_printed_text(self, image):
        """Specialized preprocessing for clear printed text"""
        img = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(self.debug_dir, "5_printed_text.jpg"), binary)
        return binary

    def recognize_with_easyocr(self, image, preprocessing_method=None):
        """Recognize text using EasyOCR with optional preprocessing"""
        if not EASYOCR_AVAILABLE:
            return None, 0, f"easyocr_{preprocessing_method}_not_available"

        try:
            # Apply preprocessing
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

            print(f"EasyOCR ({preprocessing_method}): '{full_text}' (Conf: {avg_confidence:.2f}%)")

            return full_text, avg_confidence, f"easyocr_{preprocessing_method}"

        except Exception as e:
            print(f"Error with EasyOCR ({preprocessing_method}): {e}")
            return None, 0, f"easyocr_{preprocessing_method}_error"

    def recognize_with_tesseract(self, image, preprocessing_method=None):
        """Recognize text using Tesseract with optional preprocessing"""
        if not TESSERACT_AVAILABLE:
            return None, 0, f"tesseract_{preprocessing_method}_not_available"

        try:
            # Apply the same preprocessing methods
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

            # Tesseract configuration parameters
            config = '--oem 1 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"'

            # Run Tesseract
            text = pytesseract.image_to_string(processed_image, config=config)
            text = text.strip()

            # Find JD references
            jd_pattern = r'JD[A-Z0-9\-]+'
            matches = re.findall(jd_pattern, text)

            if not matches:
                return None, 0, f"tesseract_{preprocessing_method}_no_jd"

            full_text = ' '.join(matches)
            # Tesseract doesn't provide confidence, use a fixed value
            confidence = 70

            print(f"Tesseract ({preprocessing_method}): '{full_text}' (Conf: {confidence:.2f}%)")

            return full_text, confidence, f"tesseract_{preprocessing_method}"

        except Exception as e:
            print(f"Error with Tesseract ({preprocessing_method}): {e}")
            return None, 0, f"tesseract_{preprocessing_method}_error"

    def recognize_with_paddleocr(self, image, preprocessing_method=None):
        """Recognize text using PaddleOCR with optional preprocessing"""
        if not PADDLE_AVAILABLE:
            return None, 0, f"paddle_{preprocessing_method}_not_available"

        try:
            # Apply the same preprocessing methods
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

            # Save preprocessed image to a temporary file (PaddleOCR works better with files)
            temp_path = os.path.join(self.debug_dir, f"paddle_temp_{preprocessing_method}.jpg")
            cv2.imwrite(temp_path, processed_image)

            # Run PaddleOCR
            results = self.paddle_reader.ocr(temp_path, cls=True)

            if not results or len(results) == 0 or len(results[0]) == 0:
                return None, 0, f"paddle_{preprocessing_method}_notext"

            texts = []
            confidences = []

            for line in results[0]:
                text = line[1][0].strip()
                conf = line[1][1] * 100  # Convert to percentage

                if text and text.startswith("JD"):
                    texts.append(text)
                    confidences.append(conf)

            if not texts:
                return None, 0, f"paddle_{preprocessing_method}_no_jd"

            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            print(f"PaddleOCR ({preprocessing_method}): '{full_text}' (Conf: {avg_confidence:.2f}%)")

            return full_text, avg_confidence, f"paddle_{preprocessing_method}"

        except Exception as e:
            print(f"Error with PaddleOCR ({preprocessing_method}): {e}")
            return None, 0, f"paddle_{preprocessing_method}_error"

    def recognize_text(self, image):
        """Recognize text using multiple OCR engines with multiple preprocessing strategies"""
        all_results = []
        preprocessing_methods = [None, "standard", "enhanced", "inverse", "adaptive", "printed"]

        print("\n===== RUNNING ALL OCR ENGINES =====")

        # EasyOCR results
        easyocr_results = []
        for method in preprocessing_methods:
            result = self.recognize_with_easyocr(image, method)
            if result[0]:  # If text was detected
                easyocr_results.append(result)
                all_results.append(result)

        # Tesseract results
        tesseract_results = []
        if TESSERACT_AVAILABLE:
            for method in preprocessing_methods:
                result = self.recognize_with_tesseract(image, method)
                if result[0]:
                    tesseract_results.append(result)
                    all_results.append(result)

        # PaddleOCR results
        paddle_results = []
        if PADDLE_AVAILABLE:
            for method in preprocessing_methods:
                result = self.recognize_with_paddleocr(image, method)
                if result[0]:
                    paddle_results.append(result)
                    all_results.append(result)

        # Log summary of results by engine
        print("\n===== OCR RESULTS SUMMARY =====")
        print(f"EasyOCR: {len(easyocr_results)} valid results")
        print(f"Tesseract: {len(tesseract_results)} valid results")
        print(f"PaddleOCR: {len(paddle_results)} valid results")
        print(f"Total valid results: {len(all_results)}")

        if not all_results:
            return "No text detected", 0, "no_valid_results"

        # Strategy 1: Select result with highest confidence
        best_by_confidence = max(all_results, key=lambda x: x[1])
        print(
            f"\nBest by confidence: '{best_by_confidence[0]}' ({best_by_confidence[2]}) - {best_by_confidence[1]:.2f}%")

        # Strategy 2: Find most common text across all OCR engines
        texts = [r[0] for r in all_results]

        # Create similarity groups
        similarity_groups = []
        used_indices = set()

        for i, text1 in enumerate(texts):
            if i in used_indices:
                continue

            group = [i]
            used_indices.add(i)

            for j, text2 in enumerate(texts):
                if j in used_indices or i == j:
                    continue

                similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                if similarity > 0.7:  # Texts are similar enough
                    group.append(j)
                    used_indices.add(j)

            similarity_groups.append(group)

        # Find largest group
        largest_group = max(similarity_groups, key=len)

        # Find result with highest confidence in largest group
        best_in_group = max([all_results[i] for i in largest_group], key=lambda x: x[1])
        print(f"Most consistent text: '{best_in_group[0]}' ({best_in_group[2]}) - {best_in_group[1]:.2f}%")
        print(f"Found in {len(largest_group)} results across engines")

        # Strategy 3: Check for agreement across different engines
        engine_votes = {}

        for result in all_results:
            text = result[0]
            engine = result[2].split('_')[0]  # Extract engine name

            # Add or update vote for this text by this engine
            if text not in engine_votes:
                engine_votes[text] = set()
            engine_votes[text].add(engine)

        # Find text with most engine votes
        most_votes = 0
        best_by_engines = None

        for text, engines in engine_votes.items():
            if len(engines) > most_votes:
                most_votes = len(engines)
                best_by_engines = text

        if most_votes > 1:  # Text recognized by multiple engines
            # Find result with this text and highest confidence
            matching_results = [r for r in all_results if r[0] == best_by_engines]
            best_engine_agreement = max(matching_results, key=lambda x: x[1])
            print(f"Engine agreement: '{best_engine_agreement[0]}' - agreed by {most_votes} engines")

            # If more than one engine agrees, this is our best result
            if most_votes >= 2:
                return best_engine_agreement

        # Between consistent results and highest confidence, prefer consistency
        if len(largest_group) > 2:
            return best_in_group
        else:
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