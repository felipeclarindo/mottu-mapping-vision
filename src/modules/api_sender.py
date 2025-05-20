import requests


class ApiSender:
    """
    A class to send data to an API.
    """

    def __init__(self):
        """
        Initializes the ApiSender with default values.
        """
        self.api_url = ""
        self.motos = []

    def send(self) -> str:
        """
        Send the detected motorcycles data to the API.

        Args:
            detecteds_motos (list): List of detected motorcycles.

        Returns:
            str: The response from the API.
        """
        senderMoto = self.send_motos()
        senderSectors = self.send_sectors()
        senderPatio = self.send_patio()

        if senderMoto:
            print("Motos send with success!")
        if senderPatio:
            print("Patio send with success!")
        if senderSectors:
            print("Sectors send with success!")

    def send_motos(self, motos: list) -> bool:
        """
        Send the detected motorcycles data to the API.

        Args:
            motos (list): List of detected motorcycles.

        Returns:
            str: The response from the API.
        """
        pass

    def send_sectors(self, detected_sectors: dict) -> bool:
        """
        Send the sectors data to the API.

        Args:
            sectors (dict): Dictionary of sectors.

        Returns:
            str: The response from the API.
        """
        pass

    def send_patio(self, patio: str) -> bool:
        """
        Send the patio data to the API.

        Args:
            patio (str): The patio data.

        Returns:
            str: The response from the API.
        """
        pass

    def set_motos(self, motos: list) -> None:
        """
        Set the detected motorcycles data.

        Args:
            motos (list): List of detected motorcycles.
        """
        self.motos = motos

    def get_motos(self) -> list[dict]:
        """
        Get the detected motorcycles data.

        Returns:
            list: List of detected motorcycles.
        """
        return self.motos
