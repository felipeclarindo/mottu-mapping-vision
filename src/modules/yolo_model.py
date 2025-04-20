import torch
import numpy as np


class YoloModel:
    """
    A class to represent a YOLO model.
    """

    def __init__(self) -> None:
        """
        Initializes and loads the YOLO model.
        """
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the YOLO model.

        Returns:
            torch.nn.Module: The loaded YOLO model.
        """
        try:
            # Carrega o modelo yolo mais leve (você pode trocar para yolov5m ou yolov5l se quiser mais precisão)
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
            model.eval()  # Garante que está em modo de inferência
            return model
        except (RuntimeError, OSError) as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None

    def __call__(self, image: np.ndarray):
        """
        Permite que a instância seja chamada como função.

        Args:
            image (np.ndarray): Imagem em formato array (BGR ou RGB).

        Returns:
            results: Resultado da inferência.
        """
        if self.model is None:
            raise ValueError("Modelo não carregado.")

        return self.model(image)

    def predict(self, image_path: str):
        """
        Executa a predição com o caminho da imagem.

        Args:
            image_path (str): Caminho da imagem.

        Returns:
            results: Resultado da inferência.
        """
        if self.model is None:
            raise ValueError("Modelo não carregado.")

        return self.model(image_path)

    def show_results(self, results):
        """
        Exibe os resultados da predição.

        Args:
            results: Resultado da inferência.
        """
        results.show()
