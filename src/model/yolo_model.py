import torch
import numpy as np
from typing import Union


class YoloModel:
    """
    A class to represent a YOLO model.
    """

    def __init__(self) -> None:
        """
        Initializes and loads the YOLO model.
        """
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module | None:
        """
        Loads the YOLO model.

        Returns:
            torch.nn.Module: The loaded YOLO model.
        """
        try:
            model = torch.hub.load(
                "ultralytics/yolov5", "yolov5l", pretrained=True, force_reload=True
            )
            model.eval()  # Ensures it is in inference mode
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

    def start(self, frame: str | np.ndarray) -> torch.Tensor:
        """
        Start the YOLO model inference.

        Args:
            frame (str | np.ndarray): image or video frame.

        Raises:
            ValueError: If the model is not loaded.

        Returns:
            torch.Tensor: The inference results.
        """

        if self.model is None:
            raise ValueError("Model not loaded.")

        return self.model(frame)
