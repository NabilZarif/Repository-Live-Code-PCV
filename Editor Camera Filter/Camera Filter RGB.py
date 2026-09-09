import cv2
import numpy as np

# Membuka kamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

# Mode awal
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

    # Membaca frame dari kamera
    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca kamera!")
        break

    # Membuat salinan frame
    output = frame.copy()

    # ==========================================
    # FILTER MERAH
    # ==========================================

    if mode == "red":

        # Buat gambar hitam
        output = np.zeros_like(frame)

        # Ambil channel merah
        output[:, :, 2] = frame[:, :, 2]

    # ==========================================
    # FILTER HIJAU
    # ==========================================

    elif mode == "green":

        output = np.zeros_like(frame)

        # Ambil channel hijau
        output[:, :, 1] = frame[:, :, 1]

    # ==========================================
    # FILTER BIRU
    # ==========================================

    elif mode == "blue":

        output = np.zeros_like(frame)

        # Ambil channel biru
        output[:, :, 0] = frame[:, :, 0]

    # ==========================================
    # NORMAL
    # ==========================================

    elif mode == "normal":

        output = frame

    # Tampilkan mode di layar
    cv2.putText(
        output,
        f"Mode: {mode.upper()}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Tampilkan kamera
    cv2.imshow("Camera Color Filter", output)

    # Membaca tombol keyboard
    key = cv2.waitKey(1) & 0xFF

    # R = Merah
    if key == ord("r"):
        mode = "red"

    # G = Hijau
    elif key == ord("g"):
        mode = "green"

    # B = Biru
    elif key == ord("b"):
        mode = "blue"

    # N = Normal
    elif key == ord("n"):
        mode = "normal"

    # Q = Keluar
    elif key == ord("q"):
        break


# Matikan kamera
camera.release()

# Tutup semua window
cv2.destroyAllWindows()