import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def sliding_window(image, stride=1, kernel_size=3):
    windows = []
    print(image.shape)
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            window = image[y:y + stride, x:x + kernel_size]
            windows.append(window)
    return windows

def thresh_search(image):
    h,w = image.shape
    # calculate image histogram 
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    half = h*w //2
    sums = 0
    best_ind=0
    plt.hist(image.ravel(), 256, [0,256]); 
    plt.show()
    for i, elem in enumerate(hist):
        if sums >= half:
            best_ind = i
            break
        sums+= elem
    print(" best thresh : ", best_ind)
    _, best_img = cv2.threshold(
        image, 
        best_ind, 
        255, 
        cv2.THRESH_BINARY
    )
    return best_img

def get_vec(coor):
    y = coor[0]
    x = coor[1]
    return [x*x, y*y, x*y, x, y, 1]

def polynomial_arpox(img):
    """
    I(x,y) = ax^2 + by^2 + cxy + dx + ey + f 
    
    (x1^2 y1^2 x1y1 x1 y1 1) (a b c d e f)^T = I(x,y) 픽셀값
    """
    # 1. get dataset 
    w, h = img.shape
    dataset = []
    zs = [] 
    for i in range(w):
        for j in range(h):
            v = get_vec([i,j])
            dataset.append(v)
            zs.append(img[i][j])
    dataset = np.array(dataset)

    # 2.  Ax = y 
    #     x = A*y
    zs = np.array(zs)
    pinv = np.linalg.pinv(dataset) 
    X = np.matmul(pinv, zs)
    print("Estimated param : ", X)
    return X

def inference_polynomial(X, i, j):
    a = X[0]
    b = X[1]
    c = X[2]
    d = X[3]
    e = X[4]
    f = X[5]
    #I(x,y) = ax^2 + by^2 + cxy + dx + ey + f 
    return a*i*i + b*j*j + c*i*j + d*i + e*j + f

def main():
    print("Image Binarization using Least Squares")
    img = cv2.imread('hw11_sample.png', cv2.IMREAD_GRAYSCALE)
    # Q1.1
    fixed_thesh = thresh_search(img)
    cv2.imshow("Best Fixed thresh image", fixed_thesh)
    cv2.waitKey(0)
    # Q1.2
    X = polynomial_arpox(img)
    h,w =img.shape
    background = np.zeros([h,w])
    print(background.shape)
    for i in range(h):
        for j in range(w):
            v = inference_polynomial(X, j, i)
            background[i][j] = int(v)
    cv2.imshow("Estimated Background", background/255)
    cv2.waitKey(0)
    # Q1.3
    sub = img - background 
    cv2.imshow("Subtracted Background", sub)
    cv2.waitKey(0)


        
    


if __name__ == "__main__":
    main()