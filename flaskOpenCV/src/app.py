import face_detector
import determine_weather
#import cv2
import os
import shutil
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/face_detection')
def face_detection():
    return render_template("face_detection.html")

@app.route('/weather_detection')
def weather_detection():
    return render_template("weather_detection.html")

@app.route('/upload_face', methods=['post'])
def upload_face():
    target = os.path.join(APP_ROOT, 'images/')

    shutil.rmtree(target)

    if not os.path.isdir(target):
        os.mkdir(target)
        os.mkdir(target + "/weather_images")

    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)

    face_count = face_detector.detect_face(filename)

    included_extenstions = ['jpg', 'bmp', 'png', 'gif']
    file_names = [image_names for image_names in os.listdir('./images')
                  if any(image_names.endswith(ext) for ext in included_extenstions)]
    print(file_names)

    #image_names = os.listdir('./images')
    return render_template("face_detection.html", image_names=file_names, face_count=face_count)


@app.route('/upload_weather', methods=['post'])
def upload_weather():
    target = os.path.join(APP_ROOT, 'images/')

    shutil.rmtree(target)

    if not os.path.isdir(target):
        os.mkdir(target)
        os.mkdir(target + "/weather_images")

    for upload in request.files.getlist("file1"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)

    weather = determine_weather.read_image(filename)

    included_extenstions = ['jpg', 'bmp', 'png', 'gif']
    file_names = [image_names for image_names in os.listdir('./images')
                  if any(image_names.endswith(ext) for ext in included_extenstions)]
    print(file_names)

    #image_names = os.listdir('./images')
    return render_template("weather_detection.html", image_names=file_names, weather=weather)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True, port=8080)