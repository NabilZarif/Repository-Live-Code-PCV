import cv2
import numpy as np

# 1. Menghitung histogram secara manual
def hitung_histogram(im_gray):
    L = 256
    hist = np.zeros(L, dtype=np.int32)
    tinggi, lebar = im_gray.shape
    
    # Loop bersarang untuk mencacah tiap intensitas piksel
    for i in range(tinggi):
        for j in range(lebar):
            r = im_gray[i, j]
            hist[r] += 1
            
    return hist

# 2. Transformasi intensitas negatif menggunakan LUT manual
def transformasi_negatif(im_gray):
    L = 256
    lut = np.zeros(L, dtype=np.uint8)
    
    for r in range(L):
        lut[r] = 255 - r
        
    # Memanfaatkan pemetaan index array NumPy sebagai ganti nested loop.
    # Ini wajib untuk video agar tidak terjadi lag ekstrem,
    # namun tetap tidak memanggil fungsi transformasi bawaan paket.
    hasil = lut[im_gray] 
    return hasil

# 3. Ekualisasi histogram manual
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

# === Blok Eksekusi Akses Kamera ===
if __name__ == "__main__":
    # Menginisialisasi kamera utama (indeks 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera tidak dapat diakses.")
        exit()

    while True:
        # Menangkap frame demi frame
        ret, frame = cap.read()
        
        if not ret:
            print("Gagal menerima frame dari kamera.")
            break
            
        # Memperkecil resolusi frame. Penting karena perhitungan histogram manual 
        # (menggunakan iterasi for di Python) memakan waktu proses (CPU) yang besar.
        frame = cv2.resize(frame, (320, 240))
        
        # Konversi frame BGR ke Grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Terapkan fungsi transformasi pada frame saat ini
        frame_negatif = transformasi_negatif(frame_gray)
        frame_ekualisasi = ekualisasi_manual(frame_gray)
        
        # Tampilkan jendela hasil
        cv2.imshow('Kamera Asli (Grayscale)', frame_gray)
        cv2.imshow('Kamera - Transformasi Negatif', frame_negatif)
        cv2.imshow('Kamera - Ekualisasi Histogram', frame_ekualisasi)
        
        # Tunggu 1 ms, dan periksa apakah tombol 'q' ditekan untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan memori dan putuskan akses kamera setelah loop berhenti
    cap.release()
    cv2.destroyAllWindows()