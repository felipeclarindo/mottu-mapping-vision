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
                "ultralytics/yolov5", "yolov5m", pretrained=True, force_reload=True
            )
            model.eval()
            return model
        except (RuntimeError, OSError) as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None

    def __call__(self, image: np.ndarray):
        """
        Allows the class to be called as a function.

        Args:
            image (np.ndarray): image or video frame(rgb format).
        Returns:
            results: Resultado da inferência.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

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
