import cv2
import numpy as np

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

mode = "normal"

print("================================")
print("       CAMERA COLOR FILTER")
print("================================")
print("Tekan R = Merah")
print("Tekan G = Hijau")
print("Tekan B = Biru")
print("Tekan N = Normal")
print("Tekan Q = Keluar")
print("================================")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca kamera!")
        break

    output = frame.copy()

    if mode == "red":

        output = np.zeros_like(frame)

        output[:, :, 2] = frame[:, :, 2]

    elif mode == "green":

        output = np.zeros_like(frame)

        output[:, :, 1] = frame[:, :, 1]

    elif mode == "blue":

        output = np.zeros_like(frame)

        output[:, :, 0] = frame[:, :, 0]

    elif mode == "normal":

        output = frame

    cv2.putText(
        output,
        f"Mode: {mode.upper()}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera Color Filter", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        mode = "red"

    elif key == ord("g"):
        mode = "green"

    elif key == ord("b"):
        mode = "blue"

    elif key == ord("n"):
        mode = "normal"

    elif key == ord("q"):
        break

camera.release()

cv2.destroyAllWindows()
