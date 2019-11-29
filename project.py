# Call packages
import requests
import urllib.request
import base64
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imutils
plt.style.use('dark_background')

# Image capture from Station -> save it as photo.jpg


def StationWork():
    url = 'http://172.16.62.115:8080/photo.jpg'

    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)

    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;)
    img = cv2.imdecode(imgNp, -1)

    # Save Image as photo.jpg
    cv2.imwrite('photo.jpg', img)

    # put the image on screen
    # plt.figure(figsize=(12,10))
    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()


# Using ALPR api to get String from CAR PLATE
def plateText(IMAGE_PATH):
    # Secret key from ALPR site
    SECRET_KEY = 'sk_3ad45d755d67600b8fcb0a78'

    # Open Image
    with open(IMAGE_PATH, 'rb') as image_file:
        img_base64 = base64.b64encode(image_file.read())

    # search on
    url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=kor&secret_key=%s' % (SECRET_KEY)
    r = requests.post(url, data=img_base64)
    json_data = r.json()
    x1_co = json_data['results'][0]['coordinates'][0]['x']
    y1_co = json_data['results'][0]['coordinates'][0]['y']
    x2_co = json_data['results'][0]['coordinates'][2]['x']
    y2_co = json_data['results'][0]['coordinates'][2]['y']

    # Crop plate from Original Image
    img = Image.open(IMAGE_PATH)
    area = (x1_co, y1_co, x2_co, y2_co)
    crop = img.crop(area)
    crop.save('found_plate.jpg')


# Find Highest correlation from Original Image by EV mark template.
def FindRectangle():
    template_ori = cv2.imread('evmark2.png')
    template = cv2.cvtColor(template_ori, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    image = cv2.imread('found_plate.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.1, 2.0, 50)[::-1] :

        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, scale)
        
        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
    (_, maxLoc, r, scale) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    return cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        
def ShowMatches(img):
    MIN_MATCH_COUNT = 3

    # Template Imag
    img1 = cv2.imread('trueev2.png')

    # Cropped Image
    img2 = cv2.imread(img)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts)

        matchesMask = None
        h, w, d = img1.shape
        pts = np.float32([ [0, 0], [0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], False,255,3, cv2.LINE_AA)
    

    else:
        matchesMask = None
    print("Matches are found {}".format(len(good)))


    if len(good) < 1:
        plateIs = 'General Vehicle'
        img3 = img2
    else:
        plateIs = 'Eletronic Vehicle'

        draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                       singlePointColor= None, # draw only inliers
                       flags= 0)
        img3 = cv2.drawMatches(img1, kp1, img2,kp2,good,None,**draw_params)
#    if len(good) < 3:
#        ori_img = cv2.imread(img)
#        o_w, o_h, o_g = ori_img.shape
#        o_w, o_h, o_g = np.int(o_w/2), np.int(o_h/2), np.int(o_g/2)
#        texted_image = cv2.putText(img=np.copy(ori_img), text=plateIs, org=(o_h, o_w),fontFace=1, fontScale=5, color=(0,0,255), thickness=2) 
#    else:
#        texted_image = cv2.putText(img=np.copy(img3), text=plateIs, org=(5, 50),fontFace=1, fontScale=1, color=(0,0,255), thickness=2)

#    plt.figure(figsize=(12, 10))
#    plt.axis("off")
#    plt.imshow(cv2.cvtColor(texted_image, cv2.COLOR_BGR2RGB)), plt.show()
    cv2.imwrite('saved.jpg', img3)
    return plateIs


def realtime_plate_recog():
    StationWork()
    FindRectangle('photo.jpg')
    ShowMatches()
