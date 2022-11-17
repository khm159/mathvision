from cmath import exp
import cv2
import math
import numpy as np
import argparse
from sklearn import linear_model
import matplotlib.pyplot as plt

window_size = (640, 480)

def on_mouse(event, x, y, buttons, user_param):

    def close_get_data(points):
        if len(points) >= 2:
            # print(f"points:{points}")
            return True
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

def ransac_exp(points, iterations, linear_thresh=None, outlier_ratio=None):

    pts = []
    for pt in points:
        pts.append(np.array([pt[0], pt[1]]))
    pts = np.array(pts)

    X = np.expand_dims(pts[:,0],axis=1)
    y = np.expand_dims(pts[:,1],axis=1)
    

    # 1. Fit least-squre 
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # # Least-square estimated coefficient
    # print("Estimated coefficients (least-square)")
    # print(lr.coef_)

    # Predicted data of least-square
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    plt.title("Least-Square Result")
    plt.scatter(
        X, y, color="green", marker=".", label="input points"
    )
    plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()



    # # 2. Robustly fit linear model with RANSAC algorithm
    if linear_thresh is None and outlier_ratio is None:
        ransac = linear_model.RANSACRegressor(
            max_trials = iterations
        )
    elif linear_thresh is None:
        ransac = linear_model.RANSACRegressor(
            max_trials = iterations
        )
    elif outlier_ratio is None:
        ransac = linear_model.RANSACRegressor(
            max_trials = iterations,
            residual_threshold = linear_thresh
        )

    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    # # Compare estimated coefficients
    # print("Estimated coefficients (RANSAC):")
    # print(ransac.estimator_.coef_)

    plt.title('RANSAC')
    plt.scatter(
        X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
    )
    plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
    plt.plot(line_X, line_y_ransac, color="red", linewidth=2, label="RANSAC regressor")
    
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()


def App():
    """
    main App 
    """
    global done, points, current, prev_current, frame, mode

    done = False
    points = []
    current = (-10,-10)
    prev_current = (0,0)
    frame = np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255

    cv2.namedWindow("HW12")
    cv2.setMouseCallback("HW12", on_mouse)
    while True:
        draw_frame = frame.copy()
        if len(points) == 0:
            text = "Draw Lines : Needs more than 2 points"
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
            # EXP2 : Num iter
            exp_max_iter  = [100]
            exp_threshold = [10]
            for iter in exp_max_iter:
                for thresh in exp_threshold:
                    print("=====================")
                    print("> Iteration : ", iter)
                    print("> Thresh    : ", thresh)
                    ransac_exp(
                        points=points, 
                        iterations=iter,
                        linear_thresh=thresh
                        )

        cv2.imshow("HW12", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("HW12")

if __name__ == '__main__':

    # select option
    App()

