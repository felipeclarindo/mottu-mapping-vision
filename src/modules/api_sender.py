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

    def send(self, data) -> str:
        """
        Send the detected motorcycles data to the API.

        Args:
            detecteds_motos (list): List of detected motorcycles.

        Returns:
            str: The response from the API.
        """
        senderMoto = self.send_motos(data["motos"])
        senderSectors = self.send_sectors(data["sectors"])
        senderPatio = self.send_patio(data["patio"])

        if senderMoto:
            print("Motos send with success;")
        if senderPatio:
            print("Patio send with success;")
        if senderSectors:
            print("Sectors send with success;")

    def send_motos(self, detected_motos: list) -> bool:
        """
        Send the detected motorcycles data to the API.

        Args:
            detected_motos (list): List of detected motorcycles.

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
        pass
