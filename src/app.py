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
        """
        Run the main application logic (real-time camera feed).
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Error to access the camera.")
            return

        print("Capturing video... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = frame[..., ::-1]

            results = self.model.predict(frame_rgb)
            df = results.pandas().xyxy[0]

            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(
                    int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
                )
                label = row["name"]

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

            cv2.imshow("Detecção em tempo real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Computational Vision stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()
