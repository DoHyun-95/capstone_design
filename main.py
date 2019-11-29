import cv2
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import project
from datetime import datetime, timedelta
import math



start = datetime.now()
image = 'e23.jpg'
# calculate size of image
KB = int(math.floor(os.path.getsize(image)/1024) )

# if image size is smaller than 300KB, print this
if KB < 50:
    result = "이미지 크기가 {}KB 입니다. 높은 화질의 이미지를 올려주세요.".format( KB )
    print(result)
else: 
    project.plateText(image)
    plate_name = 'found_plate.jpg'
    plate = cv2.imread(plate_name)
    hsv_img = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV) # image color convert to hsv

    #################################
    # Blue color range -> Electronic Vehicle
    blue_mask = cv2.inRange(hsv_img, (85,80,20) , (125,255,255) )
    blue = cv2.bitwise_and(plate, plate, mask = blue_mask )
    count_blue = np.count_nonzero( blue != 0 )
    ##################################
    # print('Detected blue points {}'.format( count_blue )) # find blue points as 0
    
    
    ##################################
    # Green color -> General Vehicle
    green_mask = cv2.inRange(hsv_img, (40,80,20), (80,255,255) )
    green = cv2.bitwise_and(plate, plate, mask= green_mask )
    count_green = np.count_nonzero(green !=0 )
    ##################################
    # print('Detected Green points {}'.format( count_green ))
    
    
    ##################################
    # White color -> General Vehicle
    white_mask = cv2.inRange(hsv_img, (0,0,170) , (131, 255, 255) )
    white = cv2.bitwise_and(plate, plate, mask = white_mask)
    count_white = np.count_nonzero(white != 0)
    ##################################
    # print('Detected White points {}'.format( count_white ))
    
    
    
    ##################################
    # yellow color -> Commercial car
    yellow_mask = cv2.inRange(hsv_img, (15,80,20) , (35, 255, 255) )
    yellow = cv2.bitwise_and(plate, plate, mask = yellow_mask)
    count_yellow = np.count_nonzero(yellow != 0)
    ##################################
    # print('Detected Yellow points {}'.format( count_yellow ) )
    
    
    plate_list = [blue, green, white, yellow]
    value_list = [count_blue, count_green, count_white, count_yellow]
    
    max_value = max(value_list)
    index = value_list.index(max_value)
    car_kinds = ['Electronic Vehicle','General Vehicle', 'Commercial Vehicle']
    match_result = project.ShowMatches(plate_name)
    save = cv2.imread('saved.jpg')

    if index == 0:
        car = car_kinds[0]
        plate_img = project.FindRectangle()
        plt.subplot(211)
        plt.axis("off")
        plt.imshow(cv2.cvtColor( plate_img , cv2.COLOR_BGR2RGB))
        plt.subplot(212)
        plt.axis("off")
        plt.imshow(cv2.cvtColor( save, cv2.COLOR_BGR2RGB))
        plt.show()
            
    elif index == 1 or index == 2:
        car = car_kinds[1]
        plt.axis("off")
        plt.imshow(cv2.cvtColor( save, cv2.COLOR_BGR2RGB)) 
        plt.show()
        
    else:
        car = car_kinds[2]
        plt.axis("off")
        plt.imshow(cv2.cvtColor( save , cv2.COLOR_BGR2RGB))
        plt.show()
        
    print(car)
    # print("This Car is {}".format(car) )
    # plt.imshow( cv2.cvtColor(plate_list[index], cv2.COLOR_BGR2RGB) )
    
    print(datetime.now() - start )


# color detection from BGR -> HSV
