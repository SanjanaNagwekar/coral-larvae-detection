from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class DetectionProfile:
    mode:str
    min_dist:int
    min_radius:int
    max_radius:int
    hough_param2:int=25
    component_min_area:int=10_000
    component_max_area:int=100_000

TIMEPOINT0=DetectionProfile("dark-regions",100,20,60)
TIMEPOINT1=DetectionProfile("edge-regions",200,80,100)

def profile_for_filename(filename:str)->DetectionProfile:
    return TIMEPOINT1 if "timepoint1" in filename.lower() else TIMEPOINT0

def _component_mask(binary:np.ndarray,min_area:int,max_area:int,dilation_kernel:np.ndarray)->np.ndarray:
    num_labels,labels,stats,_=cv2.connectedComponentsWithStats(binary,8,cv2.CV_32S)
    mask=np.zeros_like(binary,dtype=np.uint8)
    for label in range(1,num_labels):
        area=stats[label,cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            mask[labels==label]=255
    return cv2.dilate(mask,dilation_kernel,iterations=5)

def build_region_mask(image:np.ndarray, profile:DetectionProfile)->np.ndarray:
    working=image.copy()
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    if profile.mode=="edge-regions":
        working[np.all(working<40,axis=2)]=255
        gray=cv2.cvtColor(working,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,50,150)
        closed=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations=2)
        dilated=cv2.dilate(closed,kernel,iterations=1)
        processed=cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel,iterations=3)
        binary=cv2.bitwise_not(processed)
    else:
        gray=cv2.cvtColor(working,cv2.COLOR_BGR2GRAY)
        _,thresholded=cv2.threshold(gray,170,255,cv2.THRESH_BINARY_INV)
        dilated=cv2.dilate(thresholded,np.ones((5,5),np.uint8),iterations=1)
        processed=cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel)
        processed=cv2.bitwise_not(processed)
        processed=cv2.morphologyEx(processed,cv2.MORPH_CLOSE,kernel,iterations=2)
        processed=cv2.morphologyEx(processed,cv2.MORPH_CLOSE,kernel,iterations=3)
        binary=cv2.bitwise_not(processed)
    return _component_mask(binary,profile.component_min_area,profile.component_max_area,kernel)

def detect_coral_larvae(image:np.ndarray, profile:DetectionProfile)->tuple[np.ndarray,np.ndarray,int]:
    mask=build_region_mask(image,profile)
    masked=cv2.bitwise_and(image,image,mask=mask)
    gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=profile.min_dist,param1=50,param2=profile.hough_param2,minRadius=profile.min_radius,maxRadius=profile.max_radius)
    annotated=image.copy()
    count=0
    if circles is not None:
        rounded=np.uint16(np.around(circles))
        count=len(rounded[0])
        for x,y,radius in rounded[0]:
            cv2.circle(annotated,(int(x),int(y)),int(radius),(0,255,0),2)
            cv2.circle(annotated,(int(x),int(y)),2,(0,0,255),3)
    return annotated,mask,count
