# import cv2 
# import numpy as np
# car_cascade = cv2.CascadeClassifier('./data/Haarcascades/haarcascade_car.xml')


# cap = cv2.VideoCapture("./data/cars.avi")
# # take the first frame of the video
# ret , frame = cap.read()
# g = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# carss = car_cascade.detectMultiScale( g, 1.1, 1)
# print(carss[0])
# # for (x,y,w,h) in carss:
# #     print(x,y,w,h)



# track_window = carss[3] 
# x,y,width,height=  carss[0] 


# roi = frame[y:y+height,x:x+width]
 
# hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi,np.array((0., 60., 32.)),np.array((180., 255., 255)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist, roi_hist,0,255, cv2.NORM_MINMAX)
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
# cv2.imshow('roi',roi)
# while(1):
#     ret , frame = cap.read()
#     if ret == True:
#         # frame = cv2.resize(frame, 520,dsize=5)
#         hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit) 
#         # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
#         # pts = cv2.boxPoints(ret)
#         # pts = np.int0(pts)
#         # img2 = cv2.polylines(frame,[pts],True, 255,2)
#         x,y,w,h = track_window
#         print(track_window)
#         final_image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) 
#         cv2.imshow('frame',final_image)
#         k = cv2.waitKey(30) & 0xFF
#         if k == 27:
#             break

 
# cap.release()
# cv2.destroyAllWindows()


import cv2 
import numpy as np
car_cascade = cv2.CascadeClassifier("./data/Haarcascades/haarcascade_car.xml")


cap = cv2.VideoCapture("./data/slow_traffic_small.mp4")
# take the first frame of the video
ret , frame = cap.read()
g = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
carss = car_cascade.detectMultiScale( g, 1.1, 1)
# print(carss)
for (x,y,w,h) in carss:
    print(x,y,w,h)
    



# track_window = carss[0] 
# print(track_window)
# x,y,width,height=  carss[0] 


# roi = frame[y:y+height,x:x+width]
 
# hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi,np.array((0., 60., 32.)),np.array((180., 255., 255)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist, roi_hist,0,255, cv2.NORM_MINMAX)
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
# cv2.imshow('roi',roi)
# while(1):
#     ret , frame = cap.read()
#     if ret == True:
#         # frame = cv2.resize(frame, 520,dsize=5)
#         hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit) 
#         # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
#         # pts = cv2.boxPoints(ret)
#         # pts = np.int0(pts)
#         # img2 = cv2.polylines(frame,[pts],True, 255,2)
#         x,y,w,h = track_window
#         print(track_window)
#         final_image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) 
#         cv2.imshow('frame',final_image)
#         k = cv2.waitKey(30) & 0xFF
#         if k == 27:
#             break

 
cap.release()
cv2.destroyAllWindows()