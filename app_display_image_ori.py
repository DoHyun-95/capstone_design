import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory,send_file
import cv2
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import project
from datetime import datetime, timedelta
import math

from io import BytesIO, StringIO
import numpy as np

__author__ = 'DoHyun'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
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

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
@app.route('/results/<filename>')
def results_image(filename):
    path = 'images/' + filename
    start = datetime.now()
    image = path
    # calculate size of image
    KB = int(math.floor(os.path.getsize(image) / 1024))

    # if image size is smaller than 300KB, print this
    if KB < 50:
        result = "이미지 크기가 {}KB 입니다. 높은 화질의 이미지를 올려주세요.".format(KB)
        print(result)
    else:
        project.plateText(image)
        plate_name = 'found_plate.jpg'
        plate = cv2.imread(plate_name)
        hsv_img = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)  # image color convert to hsv

        #################################
        # Blue color range -> Electronic Vehicle
        blue_mask = cv2.inRange(hsv_img, (85, 80, 20), (125, 255, 255))
        blue = cv2.bitwise_and(plate, plate, mask=blue_mask)
        count_blue = np.count_nonzero(blue != 0)
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

        plate_list = [blue, green, white, yellow]
        value_list = [count_blue, count_green, count_white, count_yellow]

        max_value = max(value_list)
        index = value_list.index(max_value)
        car_kinds = ['Electronic Vehicle', 'General Vehicle', 'Commercial Vehicle']
        match_result = project.ShowMatches(plate_name)
        save = cv2.imread('saved.jpg')

        if index == 0:
            car = car_kinds[0]
            plate_img = project.FindRectangle()
            plt.subplot(211)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            plt.subplot(212)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
            plt.show()

        elif index == 1 or index == 2:
            car = car_kinds[1]
            plt.axis("off")
            plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
            plt.show()

        else:
            car = car_kinds[2]
            plt.axis("off")
            plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
            plt.show()
        print(car)
        # print("This Car is {}".format(car) )
        # plt.imshow( cv2.cvtColor(plate_list[index], cv2.COLOR_BGR2RGB) )

        print(datetime.now() - start)

    return render_template("complete_display_image_results.html", image_name=filename)

    # color detection from BGR -> HSV


if __name__ == "__main__":
    app.run(port=4555, debug=True)
