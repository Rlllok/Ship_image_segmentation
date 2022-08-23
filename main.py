import cv2
import model
import matplotlib.pyplot as plt


def main():
    for x in range(1, 6):
        output = model.predict("unet.h5", f"./test_imgs/{x}.jpg")
        cv2.imwrite(f"./output_masks/{x}_output.jpg", output*255)

if __name__ == "__main__":
    main()