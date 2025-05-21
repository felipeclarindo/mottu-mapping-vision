import cv2
import numpy as np
from pathlib import Path
from ..model.yolo_model import YoloModel
from .api_sender import ApiSender
from .plate_and_sector_detector import PlateAndSectorDetector


class ComputationalVision:
    """
    A class to handle computational vision tasks such as image and video processing.
    """

    def __init__(self) -> None:
        """
        Initializes the ComputationalVision class with default values.
        """
        self.model = YoloModel()
        self.api_sender = ApiSender()
        self._detector = PlateAndSectorDetector()

    def capture_image(self) -> None:
        """
        Capture an image from a file and process it.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        img_path = str(
            Path(__file__).parent.parent
            / "samples"
            / "patio_mottu"
            / "with_plate"
            / "img1.png"
        )
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found in: {img_path}")

        self.process_frame(img, "IMAGE")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def capture_video(self) -> None:
        """
        Capture video from the camera and process each frame.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Error to access the camera.")
            return None

        print("Capturing video... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, "VIDEO")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Computational Vision stopped.")
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.array, input_type: str) -> None:
        """
        Process the frame to detect motorcycles and their plates.

        Args:
            frame (np.array): The frame to process.
            input_type (str): The type of input ("IMAGE" or "VIDEO").
        """
        frame_rgb = frame[..., ::-1]
        results = self.model.start(frame_rgb)
        df = results.pandas().xyxy[0]

        motos_detectadas = []

        # Detect motos
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(
                int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            )
            label = row["name"].lower()

            if "moto" in label:
                motos_detectadas.append(
                    {"coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}},
                )

        for moto in motos_detectadas:
            detection = self._detector.detect(moto["coordinates"], frame)
            if detection.get("sector_color") is None:
                print("No sector color detected.")

            if detection.get("plate") is None:
                print("No plate detected.")

            moto["sector_color"] = detection.get("sector_color", None)
            moto["plate"] = detection.get("plate", None)

        print(motos_detectadas)
        # Draw rectangles and text
        for moto in motos_detectadas:
            x1 = moto["coordinates"]["x1"]
            y1 = moto["coordinates"]["y1"]
            x2 = moto["coordinates"]["x2"]
            y2 = moto["coordinates"]["y2"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if moto.get("sector_color") is not None and moto.get("plate") is not None:
                text = f"{moto['plate']}\n{moto['sector_color']}"
                y_offset = y1 + 20
                for line in text.split("\n"):
                    cv2.putText(
                        frame,
                        line,
                        (x1 + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    y_offset += 20

        # Show the frame
        if input_type == "VIDEO":
            cv2.imshow("Real time detection: ", frame)
        if input_type == "IMAGE":
            cv2.imshow("Image detection: ", frame)

        # Payload to API
        payload = {
            "patio": "patio_mottu",
            "motos": motos_detectadas,
        }
        print("Payload to API:")
        print(len(payload["motos"]), "motos detectadas")
