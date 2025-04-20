import cv2
import numpy as np

from ..modules.yolo_model import YoloModel


class AppTest:
    """
    Class App to start the computational vision.
    """

    def __init__(self):
        """
        Initialize the App class.
        """
        self.model = YoloModel()

    def run(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("❌ Erro ao abrir a câmera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame[..., ::-1])
            df = results.pandas().xyxy[0]

            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(
                    int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                )
                label = row["name"]

                # Base do objeto - pequena faixa logo abaixo
                roi = frame[y2 : y2 + 10, x1:x2]

                if roi.size == 0:
                    continue

                mean_color_bgr = np.mean(roi, axis=(0, 1)).astype(int)
                b, g, r = mean_color_bgr

                # Detecção por setor com base em cor
                if r > 200 and g < 100 and b < 100:
                    setor = "Vermelho"
                elif b > 200 and g < 100 and r < 100:
                    setor = "Azul"
                elif r > 200 and g > 200 and b < 100:
                    setor = "Amarelo"
                else:
                    setor = "Desconhecido"

                # Exibição
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} - Setor: {setor}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Detecção por Setor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
