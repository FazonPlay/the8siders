# import cv2
# import numpy as np
# import os
# import time
# import difflib
# import re
#
# try:
#     import easyocr
#
#     EASYOCR_AVAILABLE = True
# except ImportError:
#     EASYOCR_AVAILABLE = False
#     print("EasyOCR not available")
#
#
# class OCRProcessor:
#     def __init__(self):
#         self.last_result = None
#         self.confidence = 0
#         self.debug_dir = "debug_images"
#         os.makedirs(self.debug_dir, exist_ok=True)
#
#         # Initialize OCR engines
#         if EASYOCR_AVAILABLE:
#             self.easyocr_reader = easyocr.Reader(['en'])
#             print("EasyOCR initialized")
#
#         print(f"Available OCR engines: EasyOCR={'✓' if EASYOCR_AVAILABLE else '✗'}")
#
#     def preprocess_image(self, image, method=None):
#         """Unified preprocessing pipeline for all methods"""
#         img = image.copy()
#
#         if method == "standard":
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                          cv2.THRESH_BINARY, 11, 2)
#         elif method == "enhanced":
#             img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             enhanced = clahe.apply(gray)
#             blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#             _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             kernel = np.ones((1, 1), np.uint8)
#             return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#         elif method == "inverse":
#             img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#             return binary_inv
#         elif method == "adaptive":
#             img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                          cv2.THRESH_BINARY, 15, 8)
#         elif method == "printed":
#             img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#             sharpened = cv2.filter2D(gray, -1, kernel)
#             denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
#             _, binary = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)
#             return binary
#         elif method == "jd_format":
#             img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#             enhanced = clahe.apply(gray)
#             kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#             sharpened = cv2.filter2D(enhanced, -1, kernel)
#             _, binary = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)
#             return binary
#         elif method == "metal_curved":
#             img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
#             enhanced = clahe.apply(gray)
#             _, mask = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY)
#             mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
#             enhanced_no_glare = cv2.inpaint(enhanced, mask, 5, cv2.INPAINT_TELEA)
#             kernel_sharpen = np.array([[-1, -1, -1],
#                                        [-1, 9, -1],
#                                        [-1, -1, -1]])
#             sharpened = cv2.filter2D(enhanced_no_glare, -1, kernel_sharpen)
#             _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             return binary
#         else:
#             return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     def enhance_contrast_for_text(self, image):
#         """Enhances contrast in potential text regions"""
#         img = image.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Find connected components
#         ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # Filter to keep only potential text areas
#         text_regions = []
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if 5 < h < 50 and w > h:  # Basic text region filtering
#                 text_regions.append((x, y, w, h))
#
#         # Process each text region with a sliding window
#         result_img = img.copy()
#         text_mask = np.zeros_like(gray)
#
#         for x, y, w, h in text_regions:
#             roi = gray[y:y + h, x:x + w]
#             # Enhance contrast in this region
#             roi = cv2.equalizeHist(roi)
#             # Copy back to result image
#             text_mask[y:y + h, x:x + w] = 255
#
#         # Apply the mask to original image
#         result = cv2.bitwise_and(img, img, mask=text_mask)
#         return result
#
#     def post_process_results(self, text):
#         """Apply regex patterns to extract and validate JD codes"""
#         if text is None:
#             return None
#
#         # Clean up the text
#         text = text.replace('\n', ' ').replace('\r', ' ')
#
#         # Look for JD codes with common patterns
#         jd_patterns = [
#             r'JD\s*[R|DZ]\d{6,}',  # JD followed by R or DZ and 6+ digits
#             r'JD\s*\d{6,}',  # JD followed by 6+ digits
#             r'[JI][D0O]\s*[R|DZ]?\d{6,}'  # Common OCR errors J/I, D/0/O
#         ]
#
#         for pattern in jd_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 result = match.group(0)
#                 # Clean up common OCR errors
#                 result = result.replace('I0', 'JD').replace('ID', 'JD')
#                 result = result.replace('J0', 'JD').replace('JO', 'JD')
#                 return result
#
#         return text  # Return original if no pattern matched
#
#     def recognize_with_easyocr(self, image, preprocessing_method=None):
#         """Recognize text using EasyOCR with optional preprocessing"""
#         if not EASYOCR_AVAILABLE:
#             return None, 0, f"easyocr_{preprocessing_method}_not_available"
#
#         start_time = time.time()
#         try:
#             processed_image = self.preprocess_image(image, preprocessing_method)
#             results = self.easyocr_reader.readtext(processed_image)
#
#             if not results:
#                 return None, 0, f"easyocr_{preprocessing_method}_notext"
#
#             jd_found = False
#             jd_fragments = []
#             other_fragments = []
#
#             # Sort results by position
#             sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
#
#             for (bbox, text, prob) in sorted_results:
#                 text = text.strip()
#
#                 # Check for JD prefix or common OCR errors
#                 if text == "JD" or text.startswith("JD "):
#                     jd_found = True
#                     jd_fragments.append((text, prob * 100, bbox))
#                 elif jd_found:
#                     other_fragments.append((text, prob * 100, bbox))
#                 elif text in ["ID", "J0", "I0", "JO"]:
#                     jd_found = True
#                     jd_fragments.append(("JD", prob * 100 * 0.9, bbox))
#                 else:
#                     other_fragments.append((text, prob * 100, bbox))
#
#             # Combine fragments to form complete JD code
#             if jd_found and jd_fragments:  # Check if jd_fragments is not empty
#                 combined_text = jd_fragments[0][0]
#                 confidences = [jd_fragments[0][1]]
#
#                 for text, conf, _ in other_fragments:
#                     if combined_text.endswith(" "):
#                         combined_text += text
#                     else:
#                         combined_text += " " + text
#                     confidences.append(conf)
#
#                 avg_confidence = sum(confidences) / len(confidences) if confidences else 0
#                 return combined_text, avg_confidence, f"easyocr_{preprocessing_method}"
#
#             return None, 0, f"easyocr_{preprocessing_method}_no_jd"
#
#         except Exception as e:
#             print(f"Error with EasyOCR ({preprocessing_method}): {str(e)}")
#             return None, 0, f"easyocr_{preprocessing_method}_error"
#
#     def recognize_text(self, image):
#         """Enhanced recognize_text method with curved text support"""
#         total_start_time = time.time()
#         all_results = []
#
#         # Add metal-specific method to preprocessing methods
#         preprocessing_methods = [None, "standard", "enhanced", "inverse",
#                                  "adaptive", "printed", "jd_format", "metal_curved"]
#
#         # Try contrast enhancement
#         enhanced_image = self.enhance_contrast_for_text(image)
#
#         # Run normal processing on both original and enhanced images
#         for img in [image, enhanced_image]:
#             for method in preprocessing_methods:
#                 method_name = method if method else "default"
#                 print(f"Trying preprocessing method: {method_name}")
#
#                 if EASYOCR_AVAILABLE:
#                     result = self.recognize_with_easyocr(img, method)
#                     if result[0]:  # Check if text was detected
#                         print(f"✓ EasyOCR with {method_name}: {result[0]} ({result[1]:.1f}%)")
#                         all_results.append(result)  # Add the full result
#
#         print(f"Total valid results: {len(all_results)}")
#         print(f"Processing time: {time.time() - total_start_time:.3f}s")
#
#         if not all_results:
#             return "No text detected", 0, "no_valid_results"
#
#         # Strategy 1: Select result with the highest confidence
#         best_by_confidence = max(all_results, key=lambda x: x[1])
#
#         # Strategy 2: Find most common text across all OCR engines
#         texts = [r[0] for r in all_results]
#         similarity_groups = []
#         used_indices = set()
#
#         for i, text1 in enumerate(texts):
#             if i in used_indices:
#                 continue
#
#             group = [i]
#             used_indices.add(i)
#
#             for j, text2 in enumerate(texts):
#                 if j in used_indices or i == j:
#                     continue
#
#                 similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
#                 if similarity > 0.7:
#                     group.append(j)
#                     used_indices.add(j)
#
#             similarity_groups.append(group)
#
#         # Ensure we have at least one group
#         if not similarity_groups:
#             # If somehow we have no groups (shouldn't happen), just use the best confidence
#             best_result = best_by_confidence
#         else:
#             largest_group = max(similarity_groups, key=len)
#             best_in_group = max([all_results[i] for i in largest_group], key=lambda x: x[1])
#
#             # Strategy 3: Check for agreement across different engines
#             engine_votes = {}
#             for result in all_results:
#                 text = result[0]
#                 engine = result[2].split('_')[0]
#
#                 if text not in engine_votes:
#                     engine_votes[text] = set()
#                 engine_votes[text].add(engine)
#
#             most_votes = 0
#             best_by_engines = None
#             for text, engines in engine_votes.items():
#                 if len(engines) > most_votes:
#                     most_votes = len(engines)
#                     best_by_engines = text
#
#             if most_votes > 1:
#                 matching_results = [r for r in all_results if r[0] == best_by_engines]
#                 best_engine_agreement = max(matching_results, key=lambda x: x[1])
#
#                 # If more than one engine agrees, this is our best result
#                 if most_votes >= 2:
#                     best_result = best_engine_agreement
#                     return self.post_process_results(best_result[0]), best_result[1], best_result[2]
#
#             # Between consistent results and highest confidence, prefer consistency if the group is substantial
#             if len(largest_group) > 2:
#                 best_result = best_in_group
#             else:
#                 best_result = best_by_confidence
#
#         # Apply post-processing to extract JD code if present
#         processed_text = self.post_process_results(best_result[0])
#
#         return processed_text, best_result[1], best_result[2]
#
#     def save_result(self, text, confidence, method, image_path=None, output_dir="results"):
#         """Save OCR result to a file"""
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = os.path.basename(image_path).split('.')[0] if image_path else str(int(time.time()))
#
#         result_file = os.path.join(output_dir, f"result_{timestamp}.txt")
#         with open(result_file, 'w') as f:
#             f.write(f"Text: {text}\n")
#             f.write(f"Confidence: {confidence:.2f}%\n")
#             f.write(f"Method: {method}\n")
#             if image_path:
#                 f.write(f"Source: {image_path}\n")
#
#         return result_file




import cv2
import numpy as np
import os
import time
import difflib
import re

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available")


class OCRProcessor:
    def __init__(self):
        self.last_result = None
        self.confidence = 0
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize OCR engines
        if EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])
            print("EasyOCR initialized")

        print(f"Available OCR engines: EasyOCR={'✓' if EASYOCR_AVAILABLE else '✗'}")

    def preprocess_image(self, image, method=None):
        """Unified preprocessing pipeline for all methods"""
        img = image.copy()

        if method == "standard":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == "enhanced":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif method == "inverse":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            return binary_inv
        elif method == "adaptive":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 15, 8)
        elif method == "printed":
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
            _, binary = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)
            return binary
        elif method == "jd_format":
            img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            _, binary = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)
            return binary
        elif method == "metal_curved":
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            _, mask = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            enhanced_no_glare = cv2.inpaint(enhanced, mask, 5, cv2.INPAINT_TELEA)
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced_no_glare, -1, kernel_sharpen)
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def enhance_contrast_for_text(self, image):
        """Enhances contrast in potential text regions"""
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find connected components
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter to keep only potential text areas
        text_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 5 < h < 50 and w > h:  # Basic text region filtering
                text_regions.append((x, y, w, h))

        # Process each text region with a sliding window
        result_img = img.copy()
        text_mask = np.zeros_like(gray)

        for x, y, w, h in text_regions:
            roi = gray[y:y + h, x:x + w]
            # Enhance contrast in this region
            roi = cv2.equalizeHist(roi)
            # Copy back to result image
            text_mask[y:y + h, x:x + w] = 255

        # Apply the mask to original image
        result = cv2.bitwise_and(img, img, mask=text_mask)
        return result

    def post_process_results(self, text):
        """Improved post-processing with specific character corrections"""
        if not text:
            return text

        # Clean up the text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        # Convert to uppercase (as per requirement - no lowercase letters)
        text = text.upper()

        # Remove any non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9 ]', '', text)

        # JD prefix corrections
        text = text.replace('I0', 'JD').replace('ID', 'JD')
        text = text.replace('J0', 'JD').replace('JO', 'JD')

        # Specific character corrections
        if '6' in text and '5' not in text:
            text = text.replace('6', '5')

        # Check for patterns at the end (likely A vs 4/1 confusion)
        if re.search(r'[41]$', text):
            # If ends with 4 or 1, check if it should be A
            text = text[:-1] + 'A'

        # Check for expected JD format with more specific patterns
        jd_patterns = [
            r'JD\s*[R|DZ]\d{6,}',  # JD followed by R or DZ and 6+ digits
            r'JD\s*\d{6,}',  # JD followed by 6+ digits
            r'[JI][D0O]\s*[R|DZ]?\d{6,}'  # Common OCR errors J/I, D/0/O
        ]

        for pattern in jd_patterns:
            match = re.search(pattern, text)
            if match:
                correct_code = match.group(0)
                # Clean up common OCR errors
                correct_code = correct_code.replace('I0', 'JD').replace('ID', 'JD')
                correct_code = correct_code.replace('J0', 'JD').replace('JO', 'JD')

                # Ensure proper spacing
                if 'JD' in correct_code and not re.match(r'JD\s', correct_code):
                    correct_code = re.sub(r'JD', 'JD ', correct_code)

                return correct_code

        return text

    def recognize_with_easyocr(self, image, preprocessing_method=None):
        """Recognize text using EasyOCR with optional preprocessing"""
        if not EASYOCR_AVAILABLE:
            return None, 0, f"easyocr_{preprocessing_method}_not_available"

        start_time = time.time()
        try:
            processed_image = self.preprocess_image(image, preprocessing_method)
            results = self.easyocr_reader.readtext(processed_image)

            if not results:
                return None, 0, f"easyocr_{preprocessing_method}_notext"

            jd_found = False
            jd_fragments = []
            other_fragments = []

            # Sort results by position
            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

            for (bbox, text, prob) in sorted_results:
                text = text.strip().upper()  # Convert to uppercase

                # Check for JD prefix or common OCR errors
                if text == "JD" or text.startswith("JD "):
                    jd_found = True
                    jd_fragments.append((text, prob * 100, bbox))
                elif jd_found:
                    other_fragments.append((text, prob * 100, bbox))
                elif text in ["ID", "J0", "I0", "JO"]:
                    jd_found = True
                    jd_fragments.append(("JD", prob * 100 * 0.9, bbox))
                else:
                    other_fragments.append((text, prob * 100, bbox))

            # Combine fragments to form complete JD code
            if jd_found and jd_fragments:  # Check if jd_fragments is not empty
                combined_text = jd_fragments[0][0]
                confidences = [jd_fragments[0][1]]

                for text, conf, _ in other_fragments:
                    if combined_text.endswith(" "):
                        combined_text += text
                    else:
                        combined_text += " " + text
                    confidences.append(conf)

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                return combined_text, avg_confidence, f"easyocr_{preprocessing_method}"

            return None, 0, f"easyocr_{preprocessing_method}_no_jd"

        except Exception as e:
            print(f"Error with EasyOCR ({preprocessing_method}): {str(e)}")
            return None, 0, f"easyocr_{preprocessing_method}_error"

    def rotate_image(self, image, angle):
        """Rotate image by specified angle"""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def try_orientation(self, image, angle):
        """Try processing the image at a specific orientation angle"""
        print(f"Trying orientation: {angle}°")

        # Rotate the image if needed
        if angle == 0:
            rotated_image = image
        else:
            rotated_image = self.rotate_image(image, angle)

        # Process with a subset of methods for speed
        preprocessing_methods = [None, "standard", "enhanced", "inverse"]
        all_results = []

        for method in preprocessing_methods:
            method_name = method if method else "default"
            result = self.recognize_with_easyocr(rotated_image, method)

            if result[0]:  # If text was detected
                print(f"✓ Detected at {angle}°: {result[0]} ({result[1]:.1f}%)")
                all_results.append(result)

        # Return best result for this orientation
        if all_results:
            best_result = max(all_results, key=lambda x: x[1])
            return best_result, angle

        return (None, 0, f"no_text_at_{angle}"), angle

    def recognize_text(self, image):
        """Main OCR method with orientation detection and improved processing"""
        total_start_time = time.time()

        # Try different orientations to handle rotated text
        angles = [0, 90, 180, 270]  # 0=normal, 90=rotated right, 180=upside down, 270=rotated left
        orientation_results = []

        # Confidence threshold to consider a result "good enough" to stop rotation checks
        GOOD_CONFIDENCE_THRESHOLD = 75  # Adjust this as needed

        # First try all orientations with limited preprocessing
        for angle in angles:
            result, angle = self.try_orientation(image, angle)
            if result[0]:  # If text found
                orientation_results.append((result, angle))
                print(f"Text found at {angle}° with confidence {result[1]:.1f}%")

                # Stop checking more angles if we found a good result
                if result[1] > GOOD_CONFIDENCE_THRESHOLD:
                    print(f"Found good orientation at {angle}° - skipping remaining angles")
                    break
            else:
                print(f"No text detected at {angle}°")

        # If we found text in any orientation, use the best one
        if orientation_results:
            best_orientation = max(orientation_results, key=lambda x: x[0][1])
            best_result, best_angle = best_orientation

            print(f"Best orientation detected: {best_angle}°")

            # If confidence is already high, return this result
            if best_result[1] > 85:
                processed_text = self.post_process_results(best_result[0])
                print(f"Total processing time: {time.time() - total_start_time:.3f}s")
                return processed_text, best_result[1], f"{best_result[2]}_{best_angle}deg"

            # Otherwise, proceed with more detailed processing on the best orientation
            if best_angle != 0:
                image = self.rotate_image(image, best_angle)

        # Proceed with detailed processing on the properly oriented image
        all_results = []
        preprocessing_methods = [None, "standard", "enhanced", "inverse",
                                 "adaptive", "printed", "jd_format", "metal_curved"]
        # First pass - try all methods with original image
        print("Processing with original image...")
        for method in preprocessing_methods:
            method_name = method if method else "default"
            print(f"Trying preprocessing method: {method_name}")

            result = self.recognize_with_easyocr(image, method)
            if result[0]:  # If text was detected
                print(f"✓ {method_name}: {result[0]} ({result[1]:.1f}%)")
                all_results.append(result)

        # Second pass - only if needed with enhanced contrast
        if not all_results or max(result[1] for result in all_results) < 70:
            print("Trying enhanced contrast image...")
            enhanced_image = self.enhance_contrast_for_text(image)

            # Only try the methods that are most likely to work
            for method in preprocessing_methods[:4]:  # Use the first few methods
                method_name = method if method else "default"
                print(f"Trying preprocessing method: {method_name} (enhanced)")

                result = self.recognize_with_easyocr(enhanced_image, method)
                if result[0]:
                    print(f"✓ {method_name} (enhanced): {result[0]} ({result[1]:.1f}%)")
                    all_results.append(result)

        print(f"Total valid results: {len(all_results)}")
        print(f"Processing time: {time.time() - total_start_time:.3f}s")

        if not all_results:
            return "No text detected", 0, "no_valid_results"

        # Select the best result (highest confidence)
        best_result = max(all_results, key=lambda x: x[1])

        # Apply post-processing to improve accuracy
        processed_text = self.post_process_results(best_result[0])

        return processed_text, best_result[1], best_result[2]

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


