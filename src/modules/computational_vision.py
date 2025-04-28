import cv2
import numpy as np
from pathlib import Path
from ..model.yolo_model import YoloModel
from .api_sender import ApiSender
from .plate_detection import PlateDetector
from .sector_detector import SectorDetector


class ComputationalVision:
    """
    A class to handle computational vision tasks such as image and video processing.
    """

    def __init__(self):
        self.model = YoloModel()
        self.api_sender = ApiSender()
        self.sectors_defined = SectorDetector()
        self.plate_detector = PlateDetector()

    def capture_image(self) -> None:
        img_path = str(
            Path(__file__).parent.parent
            / "samples"
            / "patio-mottu"
            / "plate"
            / "img1.png"
        )
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found in: {img_path}")

        self.process_frame(img, "IMAGE")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
c
    def capture_video(self) -> None:
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

        motos_detectadas = []
        placas_detectadas = []

        # Detect motos and plates from YOLO results
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(
                int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            )
            label = row["name"].lower()

            if "moto" in label:
                motos_detectadas.append(
                    {"coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}},
                )
            elif "plate" in label:
                placas_detectadas.append(
                    {"coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}},
                )

        # Associar placas às motos com base na proximidade
        for moto in motos_detectadas:
            moto_center_x = (moto["coordenadas"]["x1"] + moto["coordenadas"]["x2"]) // 2
            moto_center_y = (moto["coordenadas"]["y1"] + moto["coordenadas"]["y2"]) // 2

            closest_placa = None
            closest_distance = float("inf")

            for placa in placas_detectadas:
                placa_center_x = (
                    placa["coordenadas"]["x1"] + placa["coordenadas"]["x2"]
                ) // 2
                placa_center_y = (
                    placa["coordenadas"]["y1"] + placa["coordenadas"]["y2"]
                ) // 2

                distance = (
                    (moto_center_x - placa_center_x) ** 2
                    + (moto_center_y - placa_center_y) ** 2
                ) ** 0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_placa = placa

            if closest_placa:
                placa_img = frame[
                    closest_placa["coordenadas"]["y1"] : closest_placa["coordenadas"][
                        "y2"
                    ],
                    closest_placa["coordenadas"]["x1"] : closest_placa["coordenadas"][
                        "x2"
                    ],
                ]
                placa_texto = self.plate_detector.detect(placa_img)

                if placa_texto != "Not Detected":
                    moto["placa"] = placa_texto
                else:
                    moto["placa"] = "Not Exists"
            else:
                moto["placa"] = "Not Exists"

        # Associar setores às motos com base nas coordenadas
        for moto in motos_detectadas:
            moto["setor"] = self.sectors_defined.detect_sector(
                moto["coordenadas"], frame  # Passando o frame também
            )

        # Desenhar motos, placas e setores na imagem ou vídeo
        for moto in motos_detectadas:
            x1 = moto["coordenadas"]["x1"]
            y1 = moto["coordenadas"]["y1"]
            x2 = moto["coordenadas"]["x2"]
            y2 = moto["coordenadas"]["y2"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{moto['placa']}\n{moto['setor']}"
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

        # Mostrar a imagem redimensionada
        if input_type == "VIDEO":
            cv2.imshow("Detecção em tempo real", frame)
        if input_type == "IMAGE":
            cv2.imshow("Detecção em imagem", frame)

        # Payload para API
        payload = {
            "patio": "patio_mottu",
            "motos": motos_detectadas,
        }
        print("Payload to API:")
        print(len(payload["motos"]), "motos detectadas")
