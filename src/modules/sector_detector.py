import cv2
import numpy as np


class SectorDetector:
    """Classe para definir e detectar setores com base nas cores predefinidas."""

    def __init__(self):
        """
        Inicializa a classe SectorDetector.
        """
        self.api_url = ""  # Caso queira utilizar uma API, adicione a URL aqui
        self.sectors = self.get_sectors_defined()

    def get_sectors_defined(self):
        """
        Obtém os setores, seja de uma API ou predefinidos.
        """
        return [
            {
                "name": "Setor 1",
                "color": "Azul",
                "color_rgb": [0, 0, 255],
            },
            {
                "name": "Setor 2",
                "color": "Verde",
                "color_rgb": [0, 255, 0],
            },
            {
                "name": "Setor 3",
                "color": "Vermelho",
                "color_rgb": [255, 0, 0],
            },
            {
                "name": "Setor 4",
                "color": "Amarelo",
                "color_rgb": [255, 255, 0],
            },
        ]

    def detect_sector(
        self, moto_coordinates: dict, frame: np.array, tolerance: int = 10
    ):
        """
        Detecta o setor em que a moto está localizada com base nas coordenadas e na cor do setor.

        :param moto_coordinates: Coordenadas da moto (x1, y1, x2, y2)
        :param frame: Imagem ou frame de vídeo para detecção
        :param tolerance: Margem de tolerância para as cores
        :return: Nome do setor ou "Desconhecido" se não encontrado
        """
        moto_x1, moto_y1, moto_x2, moto_y2 = (
            moto_coordinates["x1"],
            moto_coordinates["y1"],
            moto_coordinates["x2"],
            moto_coordinates["y2"],
        )

        # Converte a imagem para o espaço de cor HSV (para detecção de cores mais robusta)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Itera sobre cada setor para verificar sua cor
        for sector in self.sectors:
            sector_color = np.array(sector["color_rgb"], dtype=np.uint8)

            # Define o intervalo de cor no espaço HSV
            lower_bound = np.array(
                [sector_color[0] - tolerance, 50, 50], dtype=np.uint8
            )
            upper_bound = np.array(
                [sector_color[0] + tolerance, 255, 255], dtype=np.uint8
            )

            # Cria a máscara para a cor do setor
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

            # Verifica se algum pixel dentro da área da moto está na cor do setor
            moto_area = mask[moto_y1:moto_y2, moto_x1:moto_x2]  # Recorta a área da moto
            if np.sum(moto_area) > 0:  # Se houver algum pixel da moto na cor
                return sector["name"]

        return "Desconhecido"
