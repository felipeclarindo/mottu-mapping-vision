import torch


class YoloModel:
    """
    A class to represent a YOLO model.
    """

    def __init__(self) -> None:
        """
        Initializes the YOLO model.
        """
        self.model = None
        self.load_model()

    def load_model(self) -> torch.nn.Module:
        """
        Loads the YOLO model.

        Args:
            model_name (str): The name or path of the model to load.
            Example: 'yolov8n.pt' or 'runs/train/exp/weights/best.pt'
        Returns:
            torch.nn.Module: The loaded YOLO model.
        """
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )
            return self.model
        except (RuntimeError, OSError) as e:
            print(f"Error loading model: {e}")

    def predict(self, image_path: str):
        """
        Runs inference on an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list: Prediction results.
        """
        if self.model is None:
            raise ValueError("Model is not loaded, Check the name of the model.")

        results = self.model(image_path)
        return results

    def show_results(self, results):
        """
        Displays the results of a prediction.

        Args:
            results (list): Results from the predict() method.
        """
        for result in results:
            result.show()
