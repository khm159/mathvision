import cv2
import math
import numpy as np
import argparse
from utils import polyArea

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

def get_green(points):
    # 1. y = ax+b 모델링 
    # y1 = ax1 + b
    # y2 = ax2 + b 
    # ....
    # A 는... 
    # x1, 1 
    # x2, 1 ... 
    for i in range(len(points)):
        points[i] = list(points[i])
    points=np.array(points)
    X = points[:,0]
    Y = np.array(points[:,1])
    
    A = []
    for x in X:
        A.append([x, 1])
    A = np.array(A) 
    try:
        invA = np.linalg.inv(A)
    except:
        invA = np.linalg.pinv(A)
    # 
    params = np.matmul(invA, Y)
    #print("Param Green : ",params)
    return params

def get_red(points):
    for i in range(len(points)):
        points[i] = list(points[i])
    points=np.array(points)
    X = points[:,0]
    Y = points[:,1]
    A = np.vstack([X, Y, np.array([1]*len(Y))]).T
    a,b,c = np.linalg.svd(A)[-1][-1,:]
    return [a,b,c]


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

    cv2.namedWindow("HW10")
    cv2.setMouseCallback("HW10", on_mouse)
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
            #get_least_line(points)
            green_param = get_green(points)
            a = green_param[0]
            b = green_param[1]
            # y = ax + b 
            # 640*460
            # x = 0 
            pt1 = [0,int(a*0+b)]
            # x = 640
            pt2 = [640,int(a*640+b)]
            cv2.line(draw_frame, pt1, pt2, (0, 255, 0), 4)
            
            red_param =get_red(points)
            a = red_param[0]
            b = red_param[1]
            c = red_param[2]
            # ax + by + c = 0 

            # 640*460
            # x = 0 
            # by = -ax-c
            # y = -(a/b)x - c/b
            y0    = -(a/b)*0   - (c/b)
            y640  = -(a/b)*648 - (c/b) 
            pt1 = [0, int(y0)]
            # x = 640 
            pt2 = [640, int(y640)]
            cv2.line(draw_frame, pt1, pt2, (0, 0, 255), 4)
    


        cv2.imshow("HW10", draw_frame)
        if cv2.waitKey(50) == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyWindow("HW6")

if __name__ == '__main__':

    # select option
    App()

