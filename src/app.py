from .modules.computational_vision import ComputationalVision
from .utils.utils import clear_terminal


class App:
    """
    Class App to start the computational vision.
    """

    def __init__(self):
        """
        Initialize the App class.
        """
        self.vision = ComputationalVision()

    def show_menu_options(self) -> None:
        """
        Display the menu options.
        """
        clear_terminal()
        print("-----------------------")
        print("----- Vision Menu -----")
        print("-----------------------")
        print("Select an option:")
        print("[1] Capture video")
        print("[2] Capture image")
        print("[3] Exit")

    def run(self) -> None:
        """
        Run the main application logic
        """
        exited = False
        while not exited:
            self.show_menu_options()

            choice = input("Enter your choice: ")
            if choice == "1":
                self.vision.capture_video()
            elif choice == "2":
                self.vision.capture_image()
            elif choice == "3":
                print("Exiting...")
                exited = True
            else:
                print("Invalid choice. Please try again.")
