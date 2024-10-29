import cv2
import os

image = cv2.imread("/Users/saifal-baghdadi/python_project/fingerprint/venv/src/comGray copy.jpeg")
imgebmp= cv2.imwrite('outImge.bmp' , image)
source_image = cv2.imread('/Users/saifal-baghdadi/python_project/fingerprint/venv/database/00009_74.bmp')
score = 0
#image = None
kp1 , kp2 ,mp = None ,None, None
for file in [file for file in os.listdir('venv/database')][:800]:
    target_image = cv2.imread("venv/database/" + file)
    print(target_image)  # This should not be 'None'
    print(target_image.shape)  # This should print the dimensions of the image if loaded correctly
    sift = cv2.SIFT.create() # scal-invariant  feature transform
    kp1 ,des1 = sift.detectAndCompute(source_image , None)
    kp2 , des2 = sift.detectAndCompute(target_image , None)
    #Fast Library for Approximate Nearest Neighbors
    matches = cv2.FlannBasedMatcher(dict(algorithm = 1 , trees=10) ,dict()).knnMatch(des1 , des2 , k=2)

    mp = []
    for p , q in matches :
        if p.distance < 0.1 * q.distance :
            mp.append(p) 
            keypoint =0 
            if len(kp1) <= len(kp2):
                keypoint = len(kp1) 
            else :
                keypoint = len(kp2)
            
            if len(mp) / keypoint * 100 > score :
                score = len(mp) / keypoint * 100
                print("The best match : " + file)
                print("The score : " +str(score))
                result = cv2.drawMatches(source_image , kp1 , target_image , kp2 ,mp , None)
                result =cv2.resize(result , None , fx= 2.5 , fy= 2.5)
                cv2.imshow("result" , result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

