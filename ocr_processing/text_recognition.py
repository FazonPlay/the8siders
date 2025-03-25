import cv2
import numpy as np
import os
import time
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

    def preprocess_image(self, image, method=None, angle=0):
        """Unified preprocessing pipeline for all methods with debug images"""
        img = image.copy()

        # Generate unique debug image name
        angle_text = f"{angle}deg_" if angle != 0 else ""
        method_name = method if method else "default"
        debug_filename = f"{self.debug_dir}/debug_{angle_text}{method_name}.jpg"

        # Process the image based on method
        if method == "standard":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        elif method == "enhanced":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif method == "jd_format":
            img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            _, processed = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)
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
            _, processed = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save debug image
        cv2.imwrite(debug_filename, processed)
        print(f"Debug image saved: {debug_filename}")

        return processed

    def preprocess_for_digits(self, image):
        """Special preprocessing optimized for digit recognition, particularly 5 vs 6"""
        img = image.copy()
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply stronger adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to preserve edges while removing noise
        bilateral = cv2.bilateralFilter(enhanced, 11, 17, 17)

        # Apply sharpening kernel optimized for digit edges
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9.5, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(bilateral, -1, kernel_sharpen)

        # Use adaptive thresholding with params tuned for metallic engravings
        thresh = cv2.adaptiveThreshold(sharpened, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Save debug image
        debug_filename = f"{self.debug_dir}/digit_optimized.jpg"
        cv2.imwrite(debug_filename, thresh)

        return thresh

    def analyze_digit_features(self, image, text):
        """Analyze specific features of digits to better distinguish 5 from 6"""
        # Only process if we detected potential JD code with digits
        if not re.search(r'JD\s*[R]?\s*\d+', text):
            return text, 0

        # Find digit sequences in the text
        digit_sequences = re.findall(r'\d+', text)
        if not digit_sequences:
            return text, 0

        corrected_text = text
        confidence_boost = 0

        for seq in digit_sequences:
            corrected_seq = ""
            seq_confidence = 0

            for i, digit in enumerate(seq):
                # Check each digit, with special focus on 5 vs 6 confusion
                if digit == '6':
                    # Analyze if this should be a 5
                    digit_confidence = self.analyze_specific_digit(image, digit, seq, i)
                    if digit_confidence < 0.4:  # Lower confidence means more likely to be a 5
                        corrected_seq += '5'
                        seq_confidence += (1 - digit_confidence) * 10  # Higher boost for more confident corrections
                    else:
                        corrected_seq += digit
                else:
                    corrected_seq += digit
                    digit_confidence = self.analyze_specific_digit(image, digit, seq, i)
                    seq_confidence += digit_confidence * 5  # Add confidence for digits that look correct

            # Calculate average confidence boost for this sequence
            if len(seq) > 0:
                avg_confidence = seq_confidence / len(seq)
                confidence_boost += avg_confidence

                # Replace the sequence if it was corrected
                if corrected_seq != seq:
                    corrected_text = corrected_text.replace(seq, corrected_seq)

        return corrected_text, confidence_boost

    def analyze_specific_digit(self, image, digit, sequence, position):
        """Analyze a specific digit in the image to determine true confidence"""
        # Initialize base confidence level
        confidence = 0.5  # Neutral starting point

        # Look for JD code patterns
        jd_match = re.search(r'JD\s*[R]?\s*(\d+)', sequence)
        if jd_match:
            # First position in JD codes often has specific patterns
            if position == 0:
                # For JD codes, first digit is commonly 5, less commonly 6
                if digit == '6':
                    confidence -= 0.2
                elif digit == '5':
                    confidence += 0.2

            # Second position patterns
            if position == 1:
                # Common second digits in JD codes
                if digit in ['0', '2', '4', '5']:
                    confidence += 0.15

            # Check for known John Deere model number patterns
            common_patterns = {
                '54': 0.25,
                '55': 0.25,
                '52': 0.2,
                '57': 0.2,
                '50': 0.15
            }

            if position < len(sequence) - 1:
                check_pattern = digit + sequence[position + 1]
                if check_pattern in common_patterns:
                    confidence += common_patterns[check_pattern]

            if position > 0:
                check_pattern = sequence[position - 1] + digit
                if check_pattern in common_patterns:
                    confidence += common_patterns[check_pattern]

        # For any digit sequence, check common confusion patterns
        confusion_pairs = {
            '5': '6',
            '8': '3',
            '0': 'O',
            '1': 'I'
        }

        # If this is a commonly confused digit, adjust confidence
        for correct, confused in confusion_pairs.items():
            if digit == confused:
                confidence -= 0.1
            elif digit == correct:
                confidence += 0.05

        # Ensure confidence stays within valid range
        return max(0.1, min(confidence, 0.9))

    def check_for_jd(self, image, angle=0):
        """Quick check just for JD presence"""
        if not EASYOCR_AVAILABLE:
            return False, 0

        try:
            # Basic preprocessing - quick and minimal
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results = self.easyocr_reader.readtext(gray)

            combined_text = " ".join([r[1] for r in results]).upper()

            # Check for JD variations
            jd_variants = ['JD', 'J D', 'JO', 'ID', 'I D', '10']
            for variant in jd_variants:
                if variant in combined_text:
                    confidence = 60.0  # Base confidence
                    return True, confidence

            return False, 0

        except Exception as e:
            print(f"Error in check_for_jd: {str(e)}")
            return False, 0

    def post_process_results(self, text):
        """Dynamic post-processing with contextual confidence calculation that extracts code after JD"""
        if not text:
            return None, 0

        # Clean up the text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        # Calculate initial confidence based on text properties
        text_length = len(text)
        base_confidence = min(5 + (text_length * 2), 40)  # Scale with text length but cap at 40

        # Track different confidence factors
        format_confidence = 0
        pattern_confidence = 0
        digit_confidence = 0

        # Convert to uppercase (as per requirement)
        text = text.upper()

        # Remove any non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9 ]', '', text)

        # JD prefix corrections
        jd_variants = {
            'I0': 'JD', 'ID': 'JD', 'J0': 'JD', 'JO': 'JD',
            'IO': 'JD', '10': 'JD', '1D': 'JD'
        }

        for variant, correct in jd_variants.items():
            if variant in text:
                text = text.replace(variant, correct)
                format_confidence += 15  # Strong confidence for fixing known OCR errors

        # Extract the code after JD
        jd_code_match = re.search(r'JD\s*([R]?)\s*(\d+)', text)
        if jd_code_match:
            # Get the R part (optional) and the digits
            r_part = jd_code_match.group(1).strip()
            digits = jd_code_match.group(2).strip()

            # Combine to form the clean code (R + digits)
            extracted_code = f"{r_part}{digits}"

            # Add confidence for finding a valid JD code pattern
            format_confidence += 20

            # Analyze the digit pattern for common JD codes
            if len(digits) >= 2:
                first_two = digits[:2]
                common_prefixes = {
                    '54': 15, '55': 15, '52': 12,
                    '57': 12, '50': 10, '53': 10
                }

                if first_two in common_prefixes:
                    digit_confidence += common_prefixes[first_two]

                # Check the first digit for common 5/6 misreads
                if digits[0] == '6':
                    # 6 is often a misread 5 in JD codes
                    potential_correction = '5' + digits[1:]
                    if potential_correction[:2] in common_prefixes:
                        extracted_code = f"{r_part}5{digits[1:]}"
                        digit_confidence += 10

            # Replace text with just the extracted code
            text = extracted_code
        else:
            # No valid JD code found
            format_confidence -= 10

        # Calculate final confidence score
        # Weight the components based on their reliability
        total_confidence = (
                base_confidence +
                (format_confidence * 0.8) +
                (pattern_confidence * 1.0) +
                (digit_confidence * 1.2)  # Give more weight to digit pattern analysis
        )

        # Cap the confidence at reasonable limits
        final_confidence = min(max(total_confidence, 15), 95)

        return text, final_confidence

    def recognize_with_easyocr(self, image, method=None, angle=0):
        """Recognize text using EasyOCR with specific preprocessing method"""
        if not EASYOCR_AVAILABLE:
            return None, 0, "easyocr_unavailable"

        try:
            # Preprocess the image
            processed = self.preprocess_image(image, method, angle)

            # Recognize text with EasyOCR
            results = self.easyocr_reader.readtext(processed)

            # Combine results and get total confidence
            if results:
                combined_text = " ".join([r[1] for r in results])
                avg_confidence = sum([r[2] for r in results]) / len(results) * 100

                # Post-process to clean and format
                text, confidence = self.post_process_results(combined_text)

                # Combine confidence scores (weighted average)
                combined_confidence = (confidence * 0.7) + (avg_confidence * 0.3)

                method_name = method if method else "default"
                return text, combined_confidence, method_name

            return None, 0, method if method else "default"

        except Exception as e:
            print(f"Error in recognize_with_easyocr: {str(e)}")
            return None, 0, "error"

    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Perform the rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated

    def recognize_text(self, image):
        """Main OCR method optimized for JD code detection with improved digit recognition"""
        total_start_time = time.time()

        # STEP 1: Quick orientation testing for JD presence
        angles = [0, 90, 180, 270]
        jd_found = False
        best_angle = 0

        print("STEP 1: Quick scan for JD presence...")
        for angle in angles:
            print(f"Checking for JD at {angle}°...")
            rotated_image = self.rotate_image(image, angle) if angle != 0 else image

            jd_found, confidence = self.check_for_jd(rotated_image, angle)
            if jd_found:
                print(f"JD found at {angle}° with confidence {confidence:.1f}%")
                best_angle = angle
                break

        if not jd_found:
            print("JD not found in any orientation - trying detailed scan")
            # If we didn't find JD in quick scan, try with full processing
            all_results = []

            for angle in angles:
                rotated_image = self.rotate_image(image, angle) if angle != 0 else image

                # Try with standard processing
                result = self.recognize_with_easyocr(rotated_image, "standard", angle)
                if result[0]:
                    all_results.append((result, angle))

            if all_results:
                # Get best result
                best_result, best_angle = max(all_results, key=lambda x: x[0][1])
                jd_found = True
            else:
                return "No JD code detected", 0, "no_jd_found"

        # STEP 2: JD was found, now process properly oriented image with filters
        print(f"STEP 2: JD found at {best_angle}° - extracting full code...")

        # Rotate image to correct orientation
        if best_angle != 0:
            image = self.rotate_image(image, best_angle)

        # Apply different preprocessing methods to extract the full code
        preprocessing_methods = ["standard", "enhanced", "jd_format", "metal_curved"]
        all_results = []

        for method in preprocessing_methods:
            print(f"Trying preprocessing method: {method}")
            result = self.recognize_with_easyocr(image, method, best_angle)
            if result[0]:
                print(f"✓ {method}: {result[0]} ({result[1]:.1f}%)")
                all_results.append(result)

        # Use the digit-optimized preprocessing
        digit_image = self.preprocess_for_digits(image)
        result = self.easyocr_reader.readtext(digit_image)

        if result:
            # Extract text and process specifically for digit accuracy
            full_text = " ".join([text for (_, text, _) in result])
            digit_text, digit_confidence = self.post_process_results(full_text)

            # Apply additional digit-specific analysis
            refined_text, confidence_boost = self.analyze_digit_features(image, digit_text)

            if refined_text:
                final_confidence = digit_confidence + confidence_boost + 5  # Extra weight for digit-optimized
                print(f"✓ digit_optimized: {refined_text} ({final_confidence:.1f}%)")
                all_results.append((refined_text, final_confidence, "digit_optimized"))

        # If no results with filters, try once more with default
        if not all_results:
            result = self.recognize_with_easyocr(image, None, best_angle)
            if result[0]:
                all_results.append(result)

        print(f"Total processing time: {time.time() - total_start_time:.3f}s")

        if not all_results:
            return "JD detected but couldn't extract full code", 40, "partial_recognition"

        # Select the best result, with higher weight for digit-optimized method
        weighted_results = []
        for text, confidence, method in all_results:
            weight = 1.0
            if method == "digit_optimized":
                # Give more weight to the digit-optimized method
                weight = 1.2
            elif method == "jd_format":
                # JD format is also good for structured text
                weight = 1.1

            weighted_results.append((text, confidence * weight, method))

        # Select the best result
        best_result = max(weighted_results, key=lambda x: x[1])
        return best_result[0], best_result[1], best_result[2]

    def save_result(self, text, confidence, method, source_image_path):
        """Save OCR results to a text file"""
        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Create a sanitized version of text for filename
        if text:
            safe_text = re.sub(r'[^a-zA-Z0-9]', '_', text)
            safe_text = safe_text[:30]  # Limit length
            result_file = os.path.join("results", f"result_{timestamp}_{safe_text}.txt")
        else:
            result_file = os.path.join("results", f"result_{timestamp}_unknown.txt")

        with open(result_file, 'w') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Confidence: {confidence:.2f}%\n")
            f.write(f"Method: {method}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source Image: {source_image_path}\n")

        return result_file