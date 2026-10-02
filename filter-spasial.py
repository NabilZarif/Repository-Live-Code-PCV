# usingnamespace std
import cv2
import numpy as np

# Fungsi Unsharp Masking dengan parameter kekuatan (k)[cite: 4, 5]
def unsharp_masking(im_gray, k=1.0, ukuran=3):
    imf = im_gray.astype(np.float64) # Dihitung dalam float untuk mencegah overflow[cite: 5]
    halus = cv2.blur(imf, (ukuran, ukuran)) # Menghasilkan versi halus citra[cite: 5]
    mask = imf - halus # Mengambil bagian detail yang hilang saat dihaluskan[cite: 4, 5]
    hasil = imf + k * mask # Menambahkan detail kembali ke citra asli[cite: 5]
    return np.clip(hasil, 0, 255).astype(np.uint8) # Pemotongan rentang ke uint8[cite: 5]

def main():
    cap = cv2.VideoCapture(0)
    
    # Kernel Box (Mean) 3x3: Semua bobot bernilai 1/9 agar kecerahan rata-rata terjaga[cite: 4, 5]
    kernel_mean = np.ones((3, 3), np.float32) / 9.0
    
    # Kernel Laplacian 8-tetangga: Berjumlah nol untuk mendeteksi puncak intensitas[cite: 4, 5]
    kernel_lap = np.array([[ 1,  1,  1],
                           [ 1, -8,  1],
                           [ 1,  1,  1]], np.float32)
                           
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Penurunan resolusi untuk kelancaran video
        frame = cv2.resize(frame, (320, 240))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Filter Lowpass: Mean (Penghalus)
        hasil_mean = cv2.filter2D(frame_gray, -1, kernel_mean, borderType=cv2.BORDER_REPLICATE)
        
        # 2. Filter Lowpass: Median (Non-linear)
        hasil_median = cv2.medianBlur(frame_gray, 3)
        
        # 3. Filter Highpass: Deteksi Tepi Sobel[cite: 4, 5]
        imf = frame_gray.astype(np.float64)
        gx = cv2.Sobel(imf, cv2.CV_64F, 1, 0, ksize=3) # Turunan arah x[cite: 5]
        gy = cv2.Sobel(imf, cv2.CV_64F, 0, 1, ksize=3) # Turunan arah y[cite: 5]
        # Magnitude aproksimasi absolut untuk komputasi yang lebih murah[cite: 5]
        sobel_mag = np.clip(np.abs(gx) + np.abs(gy), 0, 255).astype(np.uint8) 
        
        # 4. Filter Highpass: Laplacian[cite: 4, 5]
        lap = cv2.filter2D(imf, -1, kernel_lap, borderType=cv2.BORDER_REPLICATE)
        hasil_laplacian = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        
        # 5. Penajaman: Unsharp Masking 
        hasil_unsharp = unsharp_masking(frame_gray, k=1.5)
        
        # Menampilkan seluruh hasil operasi ketetanggaan
        cv2.imshow('Asli (Grayscale)', frame_gray)
        cv2.imshow('Filter Mean', hasil_mean)
        cv2.imshow('Filter Median', hasil_median)
        cv2.imshow('Deteksi Tepi (Sobel)', sobel_mag)
        cv2.imshow('Detail Turunan Ke-2 (Laplacian)', hasil_laplacian)
        cv2.imshow('Penajaman (Unsharp Masking)', hasil_unsharp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()