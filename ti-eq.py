import cv2
import numpy as np

def hitung_histogram(im_gray):
    L = 256
    hist = np.zeros(L, dtype=np.int32)
    tinggi, lebar = im_gray.shape

    for i in range(tinggi):
        for j in range(lebar):
            r = im_gray[i, j]
            hist[r] += 1
            
    return hist

def transformasi_negatif(im_gray):
    L = 256
    lut = np.zeros(L, dtype=np.uint8)
    
    for r in range(L):
        lut[r] = 255 - r

    hasil = lut[im_gray] 
    return hasil

def ekualisasi_manual(im_gray):
    L = 256
    MN = im_gray.size
    
    hist = hitung_histogram(im_gray)
    
    cdf = np.zeros(L, dtype=np.float64)
    p = hist / MN
    
    cdf[0] = p[0]
    for k in range(1, L):
        cdf[k] = cdf[k - 1] + p[k]
        
    lut = np.zeros(L, dtype=np.uint8)
    for k in range(L):
        s = round((L - 1) * cdf[k])
        if s > 255: s = 255
        elif s < 0: s = 0
        lut[k] = np.uint8(s)
        
    hasil = lut[im_gray]
    return hasil

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera tidak dapat diakses.")
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Gagal menerima frame dari kamera.")
            break

        frame = cv2.resize(frame, (320, 240))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_negatif = transformasi_negatif(frame_gray)
        frame_ekualisasi = ekualisasi_manual(frame_gray)

        cv2.imshow('Kamera Asli (Grayscale)', frame_gray)
        cv2.imshow('Kamera - Transformasi Negatif', frame_negatif)
        cv2.imshow('Kamera - Ekualisasi Histogram', frame_ekualisasi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
