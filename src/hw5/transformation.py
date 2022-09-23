import cv2
import numpy as np 


def get_centroid_of_vector(vecs):
    """
    input vec (numpy array)
    """
    xs = vecs[:,0]
    ys = vecs[:,1]
    zs = vecs[:,2]
    xs_mean = np.mean(xs)
    ys_mean = np.mean(ys)
    zs_mean = np.mean(zs)
    return np.array([xs_mean, ys_mean, zs_mean])

def get_optimal_homogeneous_transformation_matrix_3d(trans1, trans2, trans3):
    """
    needs three mapping pair for dimension 3
    trans1 [pt1, pt1']
    trans1 [pt2, pt2']
    trans1 [pt3, pt3']
    """
    # 1. get optimal R 

    # body points A --> body points B 
    # RA + t = B (rigid body)
    # suppose there are noise in each body points.
    # R = Rotation movement, t = tparellel movement  
    # A --> B 
    
    A = np.array([trans1[0], trans2[0], trans3[0]])
    B = np.array([trans1[1], trans2[1], trans3[1]])
    A_bar = get_centroid_of_vector(A)
    B_bar = get_centroid_of_vector(B)
  
    # remove translation component 
    # leaving on the rotation to deal with. 
    # (A - centroid_A,)(B - centroid_B)^T

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] -= A_bar[j]

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] -= B_bar[j]
    
    sin_theta = \
        (B) / ()

    # # S = U D V^T
    # # H = (A - centroid_A,)(B - centroid_B)^T
    # # [U, S, V] = SVD(H)

    # B = np.transpose(B)
    # H = np.matmul(B)
    # U, s, V = np.linalg.svd(H, full_matrices = True)

    # # R = VU^T
    # R = np.matmul(V, np.transpose(U))

    # # 2. get t 
    # # RxA + t = B 
    # # 
    # # now we know Rotation matrix then 
    # # RXcentroid_A, + t = centroid B 
    # # t = centroid_B - Rxcentroid_A

    # t = B_bar - (np.matmul(R, A_bar))
    # return R, t
    

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

    trans1 = np.array([p1, p1_t])
    trans2 = np.array([p2, p2_t])
    trans3 = np.array([p3, p3_t])

    get_optimal_homogeneous_transformation_matrix_3d(
        trans1, trans2, trans3
    )

    # checking value is corerct 
    # Rxp4 + t = p4'
    # result = np.matmul(R, p4) + t 
    # result = np.matmul(R, p1) + t 
