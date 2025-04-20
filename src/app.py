import pandas
import cv2

from .modules.yolo_model import YoloModel


class App:
    """
    Class App to start the computational vision.
    """

    def __init__(self):
        """
        Initialize the App class.
        """
        self.model = YoloModel()

    def run(self) -> None:
        img_path = "samples/patio-mottu-example.png"
        img = cv2.imread(img_path)

        results = self.model(img[..., ::-1])

        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(
                int, row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            )
            label = row["name"]

            base_area = img[y2 - 10 : y2, x1:x2]
            mean_color = cv2.mean(base_area)[:3]
            b, g, r = map(int, mean_color)

            # Classificação básica da cor do chão
            if r > 200 and g < 100 and b < 100:
                setor = "Vermelho"
            elif b > 200 and g < 100 and r < 100:
                setor = "Azul"
            elif r > 200 and g > 200 and b < 100:
                setor = "Amarelo"
            else:
                setor = "Desconhecido"

            # Desenha bounding box e setor
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} - Setor: {setor}"
            cv2.putText(
                img,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Detecção", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
