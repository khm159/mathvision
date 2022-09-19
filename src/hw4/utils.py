from enum import Enum
import numpy as np
import cv2

class HomographyType(Enum):
    UNKNOWUN = -1
    NORMAL = 0
    CONCAVE = 1
    TWIST = 2
    REFLECTION = 3

    def __str__(self):
        return str(self.name)

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
    area = 0
    return area

