from PIL import Image
import numpy as np
from stm_alert.display.drivers.central_lcd import CentralLCD
from stm_alert.display.drivers.adjacent_lcd import AdjacentLCD
from time import sleep


class Display:
    def __init__(self, which_screen: str):
        match which_screen:
            case "central":
                self.display = CentralLCD()
                self.backlight = 100
            case "left":
                self.display = AdjacentLCD("left")
                self.backlight = 50
            case "right":
                self.display = AdjacentLCD("right")
                self.backlight = 50
            case _:
                raise ValueError(
                    f"Invalid screen. Please specify 'central', 'left', or 'right'."
                )

        # Initialize library.
        self.display.Init()
        # Clear display.
        self.display.clear()
        # Set the backlight to 100
        self.display.bl_DutyCycle(self.backlight)

    def shutdown(self):
        self.display.module_exit()

    def show(self, image: Image):
        self.display.ShowImage(image.transpose(Image.Transpose.ROTATE_90))

    @property
    def width(self):
        if isinstance(self.display, CentralLCD):
            return self.display.width
        else:
            return self.display.height

    @property
    def height(self):
        if isinstance(self.display, CentralLCD):
            return self.display.height
        else:
            return self.display.width


if __name__ == "__main__":
    displays = [Display("left"), Display("central"), Display("right")]
    images = [
        np.zeros((display.height, display.width, 3), dtype=np.uint8)
        for display in displays
    ]
    for i, (image, display) in enumerate(zip(images, displays)):
        image[0:50, :, :] = 255
        image[..., i] = 255
        image = Image.fromarray(image, "RGB")
        display.show(image)
    sleep(5)
    displays[1].show(Image.open("../examples/cat.jpg"))
    sleep(5)
    for display in displays:
        display.shutdown()
