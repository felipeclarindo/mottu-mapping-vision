import cv2
import easyocr
import numpy as np


class PlateDetector:
    """
    A class to detect plates in images.
    """

    def __init__(self):
        """
        Initializes the PlateDetector without a frame.
        """
        self.reader = easyocr.Reader(["en"])
        self.frame = None

    def detect(self, frame: np.array) -> str:
        """
        Detects a license plate from the given frame using Haar Cascades.

        Args:
            frame (np.array): The image frame to detect plates in.

        Returns:
            list: Detected plates with coordinates and plate number, or "Not Detected".
        """
        self.frame = frame

        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Load the cascade for plate detection
        plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )

        # Detect plates in the image
        plates = plate_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(plates) == 0:
            return "Not Detected"

        detected_plates = []
        for x, y, w, h in plates:
            plate_region = frame[y : y + h, x : x + w]

            # Preprocess the plate image
            plate_region_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            _, plate_region_bin = cv2.threshold(
                plate_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Use OCR to extract the text (plate number)
            ocr_result = self.reader.readtext(plate_region_bin)
            if ocr_result:
                # Get the text from the OCR result, assuming the first result is correct
                plate_number = ocr_result[0][1]
                detected_plates.append(
                    {"coordinates": (x, y, w, h), "plate_number": plate_number}
                )

        if detected_plates:
            return detected_plates
        else:
            return "Not Detected"
