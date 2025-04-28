import cv2
import easyocr
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
        self.reader = easyocr.Reader(["en"])

    def capture_image(self) -> None:
        """
        Capture an image from the camera.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
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

        motos_detectadas = []
        setores_detectados = []
        placas_detectadas = []

        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(
                int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            )
            label = row["name"].lower()

            if "setor" in label:
                setores_detectados.append(
                    {
                        "nome": label,
                        "coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )
            elif "moto" in label:
                motos_detectadas.append(
                    {
                        "coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )
            elif "placa" in label:
                placas_detectadas.append(
                    {
                        "coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )

        # Adicionar a segmentação de setores com base em cores
        setores_detectados.extend(self.detect_sectors_by_color(frame))

        # Associar placas às motos usando OCR
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

            # Se achou uma placa perto, usa OCR para reconhecer o texto; senão, coloca "Not Exists"
            if closest_placa:
                placa_img = frame[
                    closest_placa["coordenadas"]["y1"] : closest_placa["coordenadas"][
                        "y2"
                    ],
                    closest_placa["coordenadas"]["x1"] : closest_placa["coordenadas"][
                        "x2"
                    ],
                ]
                ocr_result = self.reader.readtext(placa_img)

                if ocr_result:
                    placa_text = ocr_result[0][
                        1
                    ]  # Pega o texto da primeira placa detectada
                    moto["placa"] = placa_text
                else:
                    moto["placa"] = "Not Exists"
            else:
                moto["placa"] = "Not Exists"

        # Associar motos aos setores
        for moto in motos_detectadas:
            moto_center_x = (moto["coordenadas"]["x1"] + moto["coordenadas"]["x2"]) // 2
            moto_center_y = (moto["coordenadas"]["y1"] + moto["coordenadas"]["y2"]) // 2

            setor_encontrado = "Desconhecido"
            for setor in setores_detectados:
                sx1, sy1 = setor["coordenadas"]["x1"], setor["coordenadas"]["y1"]
                sx2, sy2 = setor["coordenadas"]["x2"], setor["coordenadas"]["y2"]

                if sx1 <= moto_center_x <= sx2 and sy1 <= moto_center_y <= sy2:
                    setor_encontrado = setor["nome"]
                    break

            moto["setor"] = setor_encontrado

        # Desenhar setores detectados
        for setor in setores_detectados:
            x1, y1, x2, y2 = setor["coordenadas"].values()

            if "vermelho" in setor["nome"]:
                color = (0, 0, 255)
            elif "azul" in setor["nome"]:
                color = (255, 0, 0)
            elif "amarelo" in setor["nome"]:
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                setor["nome"],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # Desenhar motos
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

        # Payload para API
        payload = {
            "patio": "patio_mottu",
            "setores": setores_detectados,
            "motos": motos_detectadas,
        }

        # Mostrar a imagem
        if input_type == "VIDEO":
            cv2.imshow("Detecção em tempo real", frame)
        if input_type == "IMAGE":
            cv2.imshow("Detecção em imagem", frame)

        print("Payload to API:")
        print(len(motos_detectadas), "motos detectadas")
        print(len(setores_detectados), "setores detectados")

    def detect_sectors_by_color(self, frame: np.array, motos_detectadas: list) -> list:
        """
        Detect colors in the image and segment sectors based on known colors like red, blue, and yellow.
        Avoid detecting sectors in the areas where motos are detected.
        """
        setores = []
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_all = np.zeros(
            frame.shape[:2], dtype=np.uint8
        )  # Máscara para impedir detecção nas motos

        # Marcar as regiões das motos para não detectar setores lá
        for moto in motos_detectadas:
            x1, y1, x2, y2 = (
                moto["coordenadas"]["x1"],
                moto["coordenadas"]["y1"],
                moto["coordenadas"]["x2"],
                moto["coordenadas"]["y2"],
            )
            cv2.rectangle(
                mask_all, (x1, y1), (x2, y2), 255, -1
            )  # Marca a área da moto na máscara

        # Definir intervalos de cores para os setores
        colors = {
            "vermelho": ((0, 50, 50), (10, 255, 255)),
            "azul": ((90, 50, 50), (150, 255, 255)),
            "amarelo": ((15, 50, 50), (45, 255, 255)),
            "verde": ((30, 50, 50), (100, 255, 255)),
            "laranja": ((5, 50, 50), (20, 255, 255)),
            "roxo": ((130, 50, 50), (170, 255, 255)),
            "rosa": ((140, 50, 50), (170, 255, 255)),
        }

        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.bitwise_and(
                mask, cv2.bitwise_not(mask_all)
            )  # Aplica a máscara que bloqueia as áreas das motos
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filtrar áreas pequenas
                    x, y, w, h = cv2.boundingRect(contour)
                    setores.append(
                        {
                            "nome": color_name,
                            "coordenadas": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                        }
                    )

        return setores

    def api_sender(self, data: dict) -> str:
        """
        Send the data to the API.

        Args:
            data (dict): The data to send.

        Returns:
            str: The response from the API.
        """
        pass
