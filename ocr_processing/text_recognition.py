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

        if EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])
            print("EasyOCR initialized")

        print(f"Available OCR engines: EasyOCR={'✓' if EASYOCR_AVAILABLE else '✗'}")

    def preprocess_image(self, image, method=None, angle=0):
        img = image.copy()

        angle_text = f"{angle}deg_" if angle != 0 else ""
        method_name = method if method else "default"
        debug_filename = f"{self.debug_dir}/debug_{angle_text}{method_name}.jpg"

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
            kernel_sharpen = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
            sharpened = cv2.filter2D(enhanced_no_glare, -1, kernel_sharpen)
            _, processed = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(debug_filename, processed)
        print(f"Debug image saved: {debug_filename}")

        return processed

    def enhance_contrast_for_text(self, image):
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 5 < h < 50 and w > h:
                text_regions.append((x, y, w, h))

        text_mask = np.zeros_like(gray)

        for x, y, w, h in text_regions:
            roi = gray[y:y + h, x:x + w]
            text_mask[y:y + h, x:x + w] = 255

        result = cv2.bitwise_and(img, img, mask=text_mask)
        return result

    def post_process_results(self, text):
        if not text:
            return text

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.upper()
        text = re.sub(r'[^A-Z0-9 ]', '', text)
        text = text.replace('I0', 'JD').replace('ID', 'JD')
        text = text.replace('J0', 'JD').replace('JO', 'JD')

        jd_patterns = [
            r'JD\s*[R|DZ]\d{6,}',
            r'JD\s*\d{6,}',
            r'[JI][D0O]\s*[R|DZ]?\d{6,}'
        ]

        for pattern in jd_patterns:
            match = re.search(pattern, text)
            if match:
                correct_code = match.group(0)
                correct_code = correct_code.replace('I0', 'JD').replace('ID', 'JD')
                correct_code = correct_code.replace('J0', 'JD').replace('JO', 'JD')

                if 'JD' in correct_code and not re.match(r'JD\s', correct_code):
                    correct_code = re.sub(r'JD', 'JD ', correct_code)

                return correct_code

        return text

    def recognize_with_easyocr(self, image, preprocessing_method=None, angle=0):
        if not EASYOCR_AVAILABLE:
            return None, 0, f"easyocr_{preprocessing_method}_not_available"

        try:
            processed_image = self.preprocess_image(image, preprocessing_method, angle)
            results = self.easyocr_reader.readtext(processed_image)

            if not results:
                return None, 0, f"easyocr_{preprocessing_method}_notext"

            jd_found = False
            jd_fragments = []
            other_fragments = []

            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

            for (bbox, text, prob) in sorted_results:
                text = text.strip().upper()

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

            if jd_found and jd_fragments:
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
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def try_orientation(self, image, angle):
        print(f"Trying orientation: {angle}°")

        if angle == 0:
            rotated_image = image
        else:
            rotated_image = self.rotate_image(image, angle)

        preprocessing_methods = [None, "standard"]
        all_results = []

        for method in preprocessing_methods:
            method_name = method if method else "default"
            result = self.recognize_with_easyocr(rotated_image, method, angle)

            if result[0]:
                print(f"✓ Detected at {angle}°: {result[0]} ({result[1]:.1f}%)")
                all_results.append(result)

        if all_results:
            best_result = max(all_results, key=lambda x: x[1])
            return best_result, angle

        return (None, 0, f"no_text_at_{angle}"), angle

    def recognize_text(self, image):
        total_start_time = time.time()

        angles = [0, 90, 180, 270]
        orientation_results = []

        GOOD_CONFIDENCE_THRESHOLD = 70

        for angle in angles:
            result, angle = self.try_orientation(image, angle)
            if result[0]:
                orientation_results.append((result, angle))
                print(f"Text found at {angle}° with confidence {result[1]:.1f}%")

                if result[1] > GOOD_CONFIDENCE_THRESHOLD:
                    print(f"Found good orientation at {angle}° - skipping remaining angles")
                    break
            else:
                print(f"No text detected at {angle}°")

        best_angle = 0
        if orientation_results:
            best_orientation = max(orientation_results, key=lambda x: x[0][1])
            best_result, best_angle = best_orientation

            print(f"Best orientation detected: {best_angle}°")

            if best_result[1] > 85:
                processed_text = self.post_process_results(best_result[0])
                print(f"Total processing time: {time.time() - total_start_time:.3f}s")
                return processed_text, best_result[1], f"{best_result[2]}_{best_angle}deg"

            if best_angle != 0:
                image = self.rotate_image(image, best_angle)

        all_results = []
        preprocessing_methods = [None, "standard", "enhanced", "jd_format", "metal_curved"]

        print("Processing with original image...")
        for method in preprocessing_methods:
            method_name = method if method else "default"
            print(f"Trying preprocessing method: {method_name}")

            result = self.recognize_with_easyocr(image, method, best_angle)
            if result[0]:
                print(f"✓ {method_name}: {result[0]} ({result[1]:.1f}%)")
                all_results.append(result)

        if not all_results or max((result[1] for result in all_results), default=0) < 70:
            print("Trying enhanced contrast image...")
            enhanced_image = self.enhance_contrast_for_text(image)

            debug_filename = f"{self.debug_dir}/debug_{best_angle}deg_enhanced_contrast.jpg"
            cv2.imwrite(debug_filename, enhanced_image)
            print(f"Debug image saved: {debug_filename}")
            for method in preprocessing_methods[:4]:
                method_name = method if method else "default"
                print(f"Trying preprocessing method: {method_name} (enhanced)")


                result = self.recognize_with_easyocr(enhanced_image, method, best_angle)
                if result[0]:
                    print(f"✓ {method_name} (enhanced): {result[0]} ({result[1]:.1f}%)")
                    all_results.append(result)

        print(f"Total valid results: {len(all_results)}")
        print(f"Processing time: {time.time() - total_start_time:.3f}s")

        if not all_results:
            return "No text detected", 0, "no_valid_results"

        best_result = max(all_results, key=lambda x: x[1])
        processed_text = self.post_process_results(best_result[0])

        return processed_text, best_result[1], best_result[2]

    def save_result(self, text, confidence, method, image_path=None, output_dir="results"):
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