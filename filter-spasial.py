
import cv2
import numpy as np

def unsharp_masking(im_gray, k=1.0, ukuran=3):
    imf = im_gray.astype(np.float64) 
    halus = cv2.blur(imf, (ukuran, ukuran)) 
    mask = imf - halus 
    hasil = imf + k * mask 
    return np.clip(hasil, 0, 255).astype(np.uint8) 

def main():
    cap = cv2.VideoCapture(0)

    kernel_mean = np.ones((3, 3), np.float32) / 9.0

    kernel_lap = np.array([[ 1,  1,  1],
                           [ 1, -8,  1],
                           [ 1,  1,  1]], np.float32)
                           
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hasil_mean = cv2.filter2D(frame_gray, -1, kernel_mean, borderType=cv2.BORDER_REPLICATE)

        hasil_median = cv2.medianBlur(frame_gray, 3)

        imf = frame_gray.astype(np.float64)
        gx = cv2.Sobel(imf, cv2.CV_64F, 1, 0, ksize=3) 
        gy = cv2.Sobel(imf, cv2.CV_64F, 0, 1, ksize=3) 
        sobel_mag = np.clip(np.abs(gx) + np.abs(gy), 0, 255).astype(np.uint8) 

        lap = cv2.filter2D(imf, -1, kernel_lap, borderType=cv2.BORDER_REPLICATE)
        hasil_laplacian = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

        hasil_unsharp = unsharp_masking(frame_gray, k=1.5)

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
