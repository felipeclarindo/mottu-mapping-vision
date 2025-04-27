import cv2
import numpy as np
from pathlib import Path

from .yolo_model import YoloModel
from .api_sender import ApiSender


class ComputationalVision:
    """
    A class to handle computational vision tasks such as image and video processing.
    """

    def __init__(self):
        """
        Initialize the ComputationalVision class.
        """
        self.model = YoloModel()
        self.api_sender = ApiSender()

    def capture_image(self) -> None:
        """
        Capture an image from the camera.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        img_path = str(Path(__file__).parent / "samples" / "patio-mottu" / "img1.png")
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found in: {img_path}")

        self.process_frame(img, "IMAGE")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def capture_video(self) -> None:
        """
        Capture video from the camera.

        Returns:
            VideoCapture | None : The video capture object or None if dont have cam.
        """
        # Start the video capture
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
        frame_rgb = frame[..., ::-1]
        results = self.model.start(frame_rgb)

        df = results.pandas().xyxy[0]

        sectors = {"Vermelho": [], "Azul": [], "Amarelo": [], "Desconhecido": []}

        motos_detectadas = []

        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(
                int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            )
            label = row["name"]

            # Calcular a média da cor
            base_area = frame[y2 - 10 : y2, x1:x2]
            mean_color = cv2.mean(base_area)[:3]
            b, g, r = map(int, mean_color)

            if r > 200 and g < 100 and b < 100:
                setor = "Vermelho"
            elif b > 200 and g < 100 and r < 100:
                setor = "Azul"
            elif r > 200 and g > 200 and b < 100:
                setor = "Amarelo"
            else:
                setor = "Desconhecido"

            sectors[setor].append((x1, y1, x2, y2))

            # Salvar no json para api
            motos_detectadas.append(
                {
                    "label": label,
                    "setor": setor,
                    "coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

            # Desenhar a caixa da moto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Texto vertical
            text = f"{label}\nSetor: {setor}"
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

        # Draw sectors
        for setor, boxes in sectors.items():
            if not boxes:
                continue

            x1s, y1s, x2s, y2s = zip(*boxes)
            setor_x1, setor_y1 = min(x1s), min(y1s)
            setor_x2, setor_y2 = max(x2s), max(y2s)

            if setor == "Vermelho":
                color = (0, 0, 255)
            elif setor == "Azul":
                color = (255, 0, 0)
            elif setor == "Amarelo":
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)

            cv2.rectangle(frame, (setor_x1, setor_y1), (setor_x2, setor_y2), color, 2)
            cv2.putText(
                frame,
                f"Setor {setor}",
                (setor_x1, setor_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        payload = {
            "motos": motos_detectadas,
            "sectors": sectors,
            "patio": "patio_mottu",
        }

        # Send data to API
        # self.api_sender(motos_detectadas)

        if input_type == "VIDEO":
            cv2.imshow("Detecção em tempo real", frame)

        if input_type == "IMAGE":
            cv2.imshow("Detecção em imagem", frame)

    def api_sender(self, data: dict) -> str:
        """
        Send the data to the API.

        Args:
            data (dict): The data to send.

        Returns:
            str: The response from the API.
        """
        pass
