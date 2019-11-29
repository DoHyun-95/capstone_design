import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory,send_file,make_response,flash
import cv2
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import project
from datetime import datetime, timedelta
import math
import time
from io import BytesIO, StringIO
import numpy as np

### remove cache
from functools import wraps,update_wrapper
from datetime import datetime

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-control'] = 'no-store, no-cache, must-revalidate, post-check=0,pre-check =0, max-age =0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache,view)
###
__author__ = 'DoHyun'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)

    else:
        print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        ################################################################################################################
        path = 'images/' + filename
        start = datetime.now()
        image = path
        # calculate size of image
        KB = int(math.floor(os.path.getsize(image) / 1024))

        # if image size is smaller than 300KB, print this
        if KB < 30:
            result = "이미지 크기가 {}KB 입니다. 높은 화질의 이미지를 올려주세요.".format(KB)
            print(result)

        else:
            project.plateText(image)
            plate_name = 'found_plate.jpg'  # saved from platText function
            plate = cv2.imread(plate_name)
            hsv_img = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)  # image BGR color convert to hsv

            #################################
            # Blue color range -> Electronic Vehicle
            blue_mask = cv2.inRange(hsv_img, (85, 80, 20), (125, 255, 255))
            blue = cv2.bitwise_and(plate, plate, mask=blue_mask)  # if each point in range, result won't be 0.
            count_blue = np.count_nonzero(blue != 0)  # count != 0 check how many blue points in Image.
            ##################################
            # print('Detected blue points {}'.format( count_blue )) # find blue points as 0

            ##################################
            # Green color -> General Vehicle
            green_mask = cv2.inRange(hsv_img, (40, 80, 20), (80, 255, 255))
            green = cv2.bitwise_and(plate, plate, mask=green_mask)
            count_green = np.count_nonzero(green != 0)
            ##################################
            # print('Detected Green points {}'.format( count_green ))

            ##################################
            # White color -> General Vehicle
            white_mask = cv2.inRange(hsv_img, (0, 0, 170), (131, 255, 255))
            white = cv2.bitwise_and(plate, plate, mask=white_mask)
            count_white = np.count_nonzero(white != 0)
            ##################################
            # print('Detected White points {}'.format( count_white ))

            ##################################
            # yellow color -> Commercial car
            yellow_mask = cv2.inRange(hsv_img, (15, 80, 20), (35, 255, 255))
            yellow = cv2.bitwise_and(plate, plate, mask=yellow_mask)
            count_yellow = np.count_nonzero(yellow != 0)
            ##################################
            # print('Detected Yellow points {}'.format( count_yellow ) )

            plate_list = [blue, green, white, yellow]  # each plate Image list.
            value_list = [count_blue, count_green, count_white, count_yellow]  # each number of points will be list.

            max_value = max(value_list)  # Find max count in Value list
            index = value_list.index(max_value)  # Find Index in Value list
            car_kinds = ['Electronic Vehicle', 'Not Electronic Vehicle']
            match_result, detected = project.ShowMatches(plate_name)  # show Feature Matching result
            detected = match_result
            save = cv2.imread('saved.jpg')
            timestr = time.strftime("%Y%m%d-%H%M%S")
            time_data = time.strftime("%Y/%m/%d %H:%M:%S")
            savename = "results/save" + timestr + ".png"
            if detected >= 1:
                car = car_kinds[0]  # Car will be EV
                plate_img = project.FindRectangle()[0]
                plt.subplot(211)
                plt.axis("off")
                plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                plt.subplot(212)
                plt.axis("off")
                plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
                plt.savefig("static/results/save" + timestr + ".png")
                plt.show()
            else:
                if index == 0:  # if blue points detected
                    if FindRectangle()[1] == True:  # Double check from Template Matching
                        car = car_kinds[0]  # Car will be EV
                        plate_img = FindRectangle()[0]
                        plt.subplot(211)
                        plt.axis("off")
                        plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                        plt.subplot(212)
                        plt.axis("off")
                        plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
                        plt.savefig("static/results/save" + timestr + ".png")
                        plt.show()
                    else:  # Color = Blue but nothing detected from Template Matching
                        car = car_kinds[1]
                        plt.axis("off")
                        plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
                        plt.savefig("static/results/save" + timestr + ".png")
                        plt.show()

                else:  # When they found White or Green Color.
                    car = car_kinds[1]
                    plt.axis("off")
                    plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
                    plt.savefig("static/results/save" + timestr + ".png")
                    plt.show()

            processing = datetime.now() - start
            print("This Car is {}".format(car))

        # color detection from BGR -> HSV

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename, car = car, processing = processing, savename = savename, time_data = time_data)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
