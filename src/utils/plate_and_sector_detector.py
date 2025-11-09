import cv2
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import json
from os import environ
from PIL import Image
import re


class PlateAndSectorDetector:
    def __init__(self):
        """
        Initialize the PlateAndSectorDetector class.
        """
        load_dotenv()
        self._gemini_configure()

    def _gemini_configure(self):
        """
        Configures the Gemini API with the API key from environment variables.
        """
        self.api_key = environ.get("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def detect(self, moto_coordinates: dict, frame: np.array) -> dict:
        """
        Detect the color of the ground sector and license plate of the motorcycle using Gemini.

        Args:
            moto_coordinates (dict): Coordinates of the motorcycle
            frame (np.array): The image frame to analyze.

        Returns:
            dict: Detected sector color and license plate.
        """
        sectors = ["yellow", "light_green"]

        x1 = moto_coordinates["x1"]
        x2 = moto_coordinates["x2"]
        y1 = moto_coordinates["y1"]
        y2 = moto_coordinates["y2"]

        padding_bottom = 50
        padding_side = 10

        roi = frame[y1 : y2 + padding_bottom, x1 : x2 + padding_side]

        prompt = (
            "Você está vendo a imagem de uma moto. "
            "Por favor, me diga:\n"
            f"1. Qual é a **cor do setor no chão dentre dos setores: [{",".join(sectors).endswith(".")}]** onde a moto está?\n"
            "2. Qual é o **texto da placa** da moto, se estiver visível\n"
            "Responda em formato JSON com as chaves 'sector_color' e 'plate'."
            'Response Example: {"sector_color": "yellow", "plate": "ABC1234"} CASO NAO IDENTIFIQUE, RESPONDA APENAS O IDENTIFICADO E CASO NAO IDENTIFIQUE NADA, RESPONDA {}'
            "OBS: a cor sempre traga em ingles"
            "OBS: A PLACA SEMPRE TRAGA EM MAIUSCULO E SEM CARACTERES ESPECIAIS"
        )

        return self._call_llm(prompt, roi)

    def _call_llm(self, prompt: str, image_np: np.ndarray) -> dict:
        """
        Send the prompt and image to the Gemini model and process the response.

        Args:
            prompt (str): Prompt to send to the model.
            image_np (np.ndarray): Image in numpy array format.

        Returns:
            dict: Model Response.
        """
        try:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            response = self.model.generate_content(
                [prompt, pil_image],
                generation_config={"max_output_tokens": 500},
            )

            text = response.text.strip()
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    result = self._extract_json_from_text(text)
                    result = self._extract_json_from_text(text)
                    if "plate" in result and result["plate"]:
                        plate_clean = re.sub(r"[^A-Z0-9]", "", result["plate"].upper())
                        result["plate"] = plate_clean if plate_clean else None
                    if "sector_color" in result and result["sector_color"]:
                        result["sector_color"] = result["sector_color"].lower()
                    return result

        except Exception as e:
            print(f"Erro ao processar resposta do Gemini: {e}")
            return {"sector_color": None, "plate": None}

    def _extract_json_fallback(self, raw_text: str) -> dict:
        """
        Fallback to extract JSON from raw text if the main method fails.

        Args:
            raw_text (str): Response text from Gemini.

        Returns:
            dict: Extracted JSON data.
        """
        try:
            lines = raw_text.splitlines()
            sector_color, plate = None, None

            for line in lines:
                if "sector_color" in line:
                    sector_color = line.split(":")[-1].strip().strip('" ,')
                if "plate" in line:
                    plate = line.split(":")[-1].strip().strip('" ,')

            return {"sector_color": sector_color, "plate": plate}

        except Exception as e:
            print(f"Fallback falhou: {e}")
            return {"sector_color": None, "plate": None}

    def _extract_json_from_text(self, text: str) -> dict:
        """
        Extract JSON from text using regex.

        Args:
            text (str): Response text from Gemini.

        Returns:
            dict: Extracted JSON data.
        """
        try:
            # Regex para extrair o primeiro JSON do texto
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return self._extract_json_fallback(text)
        except Exception as e:
            print(f"Erro extraindo JSON com regex: {e}")
            return {"sector_color": None, "plate": None}
