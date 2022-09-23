import cv2
import math
import numpy as np 

def get_transformation(A, B):

    ## rigid body points A --> rigid body points B 

    # 1. Remove translateion componment & get rotation 
    # 1.1 find centroids 

    C_A = np.mean(A, axis=1).reshape(-1, 1)
    C_B = np.mean(B, axis=1).reshape(-1, 1)

    A_moved = A-C_A
    B_moved = B-C_B

    H = A_moved @ np.transpose(B_moved)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    # det(R) < R, reflection detected!, correcting for it
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    # 2. calculate translation t   
    t = -R @ C_A + C_B
    return R, t

if __name__ == "__main__":
    # question 
    # p1 (-0.5, 0,         2.121320) --> p1' (1.363005, -0.427130, 2.339082)
    # p2 ( 0.5, 0,         2.121320) --> p2' (1.748084,  0.437983, 2.017688)
    # p3 ( 0.5, -0.707107, 2.828427) --> p3' (2.636461,  0.184843, 2.400710)

    # p4 ( 0.5, 0.707107, 2.828427)  --> p4' (1.4981, 0.8710, 2.8837)
    # p5 (1, 1, 1) --> p5' =?   

    mappings = []
    p1   = np.array([-0.5, 0, 2.121320])
    p1_t = np.array([1.363005, -0.427130, 2.339082])
    p2   = np.array([0.5, 0, 2.121320])
    p2_t = np.array([1.748084,  0.437983, 2.017688]) 
    p3   = np.array([0.5, -0.707107, 2.828427])
    p3_t = np.array([2.636461,  0.184843, 2.400710])
    p4   = np.array([0.5, 0.707107, 2.828427])
    p4_t = np.array([1.4981, 0.8710, 2.8837])
    p5   = np.array([1, 1, 1])

    A = np.array([
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        [p1[2], p2[2], p3[2]],
    ])
    B = np.array([
        [p1_t[0], p2_t[0], p3_t[0]],
        [p1_t[1], p2_t[1], p3_t[1]],
        [p1_t[2], p2_t[2], p3_t[2]],
    ])

    R, t = get_transformation(A, B)
    A = np.array([
        [p4[0]],
        [p4[1]],
        [p4[2]]    
    ])
    B = (R@A) + t
    print(B)