from enum import Enum
import numpy as np
import cv2
import math

class HomographyType(Enum):
    UNKNOWUN = -1
    NORMAL = 0
    CONCAVE = 1
    TWIST = 2
    REFLECTION = 3

    def __str__(self):
        return str(self.name)

def get_x_y_list(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl

def shoelace_formula(x_list,y_list):
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l

def get_homography_matrix(src, dst):
    if not len(src) >= 4:
        raise ("Source points must be >= 4")
    if not len(dst) >= 4:
        raise ("Target points must be >= 4")
    if not len(src) == len(dst):
        raise ("Num of Source and target poinst should be equal")
    A = []
    b = []
    for i in range(len(src)):
        s_x, s_y = src[i]
        d_x, d_y = dst[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))

def CrossProduct(A):
    X1 = (A[1][0] - A[0][0])
    Y1 = (A[1][1] - A[0][1])
    X2 = (A[2][0] - A[0][0])
    Y2 = (A[2][1] - A[0][1])
 
    # Return cross product
    return (X1 * Y2 - Y1 * X2)

def isConvex(points):
    prev = 0
    curr = 0
    for i in range(len(points)):
        temp = [
            points[i], 
            points[(i + 1) % len(points)],
            points[(i + 2) % len(points)]
        ]
        # Update curr
        curr = CrossProduct(temp)
        # If curr is not equal to 0
        if (curr != 0):
            # If direction of cross product of
            # all adjacent edges are not same
            if (curr * prev < 0):
                return False
            else:
                # Update curr
                prev = curr
    return True

def check_twist(pts):
    """
    this is tricky.
    ans as fall as i know 
    therer are no optimum solution for this... 
    one solution i found is below.
    """
    crosses = []
    for i in range(len(pts)-1):
        pt1 = np.append(pts[i-1],0)
        pt2 = np.append(pts[i],0)
        pt3 = np.append(pts[i+1],0)
        cross = np.cross( pt2-pt1, pt3-pt1)
        crosses.append(cross)
    
    if crosses[0][2] < 0 and crosses[1][2] <0 and crosses[2][2]>0:
        return True
    else:
        return False
        
def classifyHomography(source, target):
    if len(source) != 4 or len(target) != 4: 
        return HomographyType.UNKNOWUN  

    # toward original rect 
    H = get_homography_matrix(source,target)

    # check D 
    D = H[0][0]*H[1][1] - H[0][1]*H[1][0]
    
    if not isConvex(source):
        print("> is not convex : twisted or concave")
        if check_twist(source):
            return HomographyType.TWIST
        else:
            return HomographyType.CONCAVE
    else:
        print("> is convex : normal or reflection")
        if D<0:  
            print("D < 0 ")
            return HomographyType.REFLECTION
        else:
            print("D > 0")
            return HomographyType.NORMAL

def polyArea(points):
    xy_e=get_x_y_list(points)
    area=shoelace_formula(xy_e[0],xy_e[1])
    print("- implemented : ", area)
    print("- using library : ", polyArea_geometry(points))
    return abs(area)

def polyArea_geometry(points):
    from shapely.geometry import Polygon 
    polygon = Polygon(points)
    area = polygon.area
    return area
