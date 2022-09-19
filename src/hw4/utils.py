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

def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl

def shoelace_area(x_list,y_list):
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l

def classifyHomography(pts1, pts2):
    """
    pts2 : unit
    pts1 : points 
    """
    if len(pts1) != 4 or len(pts2) != 4: 
        return HomographyType.UNKNOWUN  
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)
    return HomographyType.NORMAL

def polyArea(points):
    xy_e=explode_xy(points)
    area=shoelace_area(xy_e[0],xy_e[1])
    print("- implemented : ", area)
    print("- using library : ", polyArea_geometry(points))
    return abs(area)

def polyArea_geometry(points):
    from shapely.geometry import Polygon 
    polygon = Polygon(points)
    area = polygon.area
    return area
