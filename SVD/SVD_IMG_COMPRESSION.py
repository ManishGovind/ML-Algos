import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
def svd_compression(img, num_components=20):
    u,s,v = np.linalg.svd(img)
    print(img.shape)
    uc = u[:, :num_components]
    sc = s[:num_components]
    vc = v[:num_components, :]
    orig_size = img.shape[0] * img.shape[1]
    compressed_size = uc.shape[0]*uc.shape[1] + sc.shape[0] + vc.shape[0] * vc.shape[1]
    print("storage needed for orig data =", orig_size)
    print("storage needed for compressed data =", compressed_size, " percentage of orig size=",compressed_size/orig_size*100,"%")
    compressed_img = np.matrix(u[:, :num_components]) * np.diag(s[:num_components])* np.matrix(v[:num_components, :])
    plt.imshow(compressed_img, cmap='gray')
    plt.show()
    print(compressed_img.shape)

def main():
    image_filename = './sample.jpg'
    img = cv2.imread(image_filename,0)  
    if img is None:
        print('could not open or find the image: ', image_filename)
        exit(0)
    #cv2.imshow('Original Image', img)
    #cv2.waitKey()
    plt.imshow(img, cmap='gray')
    plt.show()
    num_components = 50
    svd_compression(img, num_components)

if __name__ == "__main__":
 sys.exit(int(main() or 0))
