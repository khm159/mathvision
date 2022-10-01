import cv2
import math
import numpy as np
import argparse
from utils import classifyHomography, polyArea

window_size = (640, 480)
mode = "circle"

def on_mouse(event, x, y, buttons, user_param):
    global mode

    def close_get_data(points):
        if mode =="circle":
            if len(points) >= 3:
                print(f"points:{points}")
                return True
        elif mode =="ellipse":
            if len(points) >= 4:
                print(f"points:{points}")
                return True         
        else:   
            print("You need to draw more points")
            return False
    
    def reset():
        global done, points, current, prev_current, frame, homography_type
        points = []
        current = (x, y)
        prev_current = (0,0)
        done = False
        homography_type = None


    global done, points, current, prev_current, frame
    if event == cv2.EVENT_MOUSEMOVE:
        if done:
            return
        current = (x, y)
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click means adding a point at current position to the list of points
        if done:
            reset()
        if prev_current == current:
            done = close_get_data(points)
            return

        # print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        points.append((x, y))
        prev_current = (x, y)
            
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Double left click means close 
        print("Double click to close")
        done = close_get_data(points)
        print("done : ", done)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means Reset 
        print("Resetting")
        reset()

def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].
    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_center_radius_from_equation(rst):
    A = rst[0]
    B = rst[1]
    C = rst[2]

    x = -A/2 
    y = -B/2
    r = math.sqrt((A*A) + (B*B) -(4*C))/2
    return x, y, r 

def get_pinv_ellipse_impl(A):
    # Singular-value decomposition
    U, s, VT = np.linalg.svd(A)
    # create m x n Sigma matrix
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
    # select
    n_elements = 1 
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # reconstruct
    B = U.dot(Sigma.dot(VT))
    # transform
    T = U.dot(Sigma)
    T = A.dot(VT.T)
    return B

def get_pinv_impl(A):
    U, s, VT = np.linalg.svd(A)
    d = 1.0 / s
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    B = VT.T.dot(D.T).dot(U.T)
    return B
        
def get_optimal_circle(points):
    """
    get optimal circle with points 
    using SVD
    """
    A = []
    result_vec = []
    for point in points:
        pt = [point[0], point[1], 1]
        A.append(pt)
        rst = -pt[0]*pt[0] - pt[1]*pt[1]
        result_vec.append(rst)
    A = np.array(A)
    result_vec = np.array(result_vec)
    
    if A.shape[0] >3 :
        inv_matrix = get_pinv_impl(A)
    elif A.shape[0] ==3:
        det = np.linalg.det(np.array(A))
        if det == 0:
            #inv_matrix = np.linalg.pinv(A)
            inv_matrix = get_pinv_impl(A)
        else:
            inv_matrix = np.linalg.inv(A)
    else:
        raise ValueError
    
    inv_matrix2 = np.linalg.pinv(A)
    print("------------")
    print("impl pinv A")
    print(inv_matrix)
    print("------------")
    print("numpy pinv")
    print(inv_matrix2)
    print("------------")
    
    rst = np.matmul(inv_matrix, result_vec)
    x,y,r = get_center_radius_from_equation(rst)
    return x, y, r


def get_optimal_ellipse(points):
    """
    get optimal ellipse with points 
    using SVD
    """
    print("get optimal ellipse")
    print("input points ", points)
    # 1. checking determinant 
    fit_ellipse(points[:,0], points[:,1])

def App(args):
    """
    main App 
    """
    global done, points, current, prev_current, frame, mode

    done = False
    points = []
    current = (-10,-10)
    prev_current = (0,0)
    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255

    cv2.namedWindow("HW6")
    cv2.setMouseCallback("HW6", on_mouse)
    mode = args.select_mode
    while True:
        draw_frame = frame.copy()
        if len(points) == 0:
            if args.select_mode =="circle":
                text = "Draw Circle : Needs more than 3 points" 
            else:
                text = "Draw Ellipse : Needs more than 4 points"
            cv2.putText(
                draw_frame, 
                "Input data points (double click: finish)" + text, 
                (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 148, 0), 1, cv2.LINE_AA
            )
        for i, point in enumerate(points):
            cv2.circle(draw_frame, point,5,(0,200,0),-1)
            cv2.putText(draw_frame, chr(65+i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if done:
            if args.select_mode == "circle":
                x,y,r = get_optimal_circle(points)
                cv2.circle(draw_frame, (int(x),int(y)), int(r), (0,0,255))
            elif args.select_mode =="ellipse":
                get_optimal_ellipse(points)

        cv2.imshow("HW6", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("HW6")

if __name__ == '__main__':

    # select option
    parser = argparse.ArgumentParser(description='HW6 implementation')
    parser.add_argument('--select_mode',
        choices=["circle", "ellipse"],
        help='Select mode : choices=["circle", "ellipse"]'
    )
    args = parser.parse_args()
    App(args)

