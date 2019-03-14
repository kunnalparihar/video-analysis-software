from flask import render_template, url_for, flash, redirect, request, session
from flaskblog import app, bcrypt
import os
from flaskblog.forms import RegistrationForm, LoginForm
# from flaskblog.models import User
from flask_login import login_user, current_user, logout_user, login_required





os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"




@app.route("/home")
def home():
    if not session.get('logged_in'):
        flash('Please login to access this page', 'info')
        return redirect(url_for('login'))
    else:
        return render_template('home.html')



'''Samarth'''

@app.route('/')
def index():
    return render_template('index.html')


'''Samarth'''
'''Vivasvan'''

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

from werkzeug.utils import secure_filename
import cv2
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from flask import send_file
import json
import sys
sys.path.insert(0, '/Users/kunal/python_code/website_flask/flaskblog/flaskblog/head_detection_using_yolo/')

from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import pandas as pd

temp1 = ""
UPLOAD_FOLDER = dir_path + '/uploads/'
print("UPLOAD", UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/return-file/")
def return_file():
    return send_file(UPLOAD_FOLDER + temp1, attachment_filename=temp1)


@app.route("/result", methods=['GET'])
def result():
    return render_template('result.html', temp1=UPLOAD_FOLDER + temp1)


@app.route("/block1", methods=['GET', 'POST'])
def block1():
    print("!#$%^&I&^%$#@!#$%^&^%$#@!@#$%^&^%$#@!@#$%^&*------------")

    if not session.get('logged_in'):
        flash('Please login to access this page', 'info')
        return redirect(url_for('login_b1'))
    else:
        global temp1
        config_path  = 'config.json'
        weights_path = 'model.h5'
        image_path   = 'image.mp4'

        with open(config_path) as config_buffer:    
            config = json.load(config_buffer)
        print("!#$%^&I&^%$#@!#$%^&^%$#@!@#$%^&^%$#@!@#$%^&*------------")

        # return render_template('result.html')
        print(config['model'])
        # yolo = YOLO(backend             = config['model']['backend'],
        #             input_size          = config['model']['input_size'], 
        #             labels              = config['model']['labels'], 
        #             max_box_per_image   = config['model']['max_box_per_image'],
        #             anchors             = config['model']['anchors'])
        print("!#$%^&I&^%$#@!#$%^&^%$#@!@#$%^&^%$#@!@#$%^&*------------")
        time_now = 0
        data_head = pd.DataFrame({'Time': [0],'Head_count':[0]})
        if request.method == 'POST':
            time_now = 0
            if request.form['btn'] == 'Upload':
                # check if the post request has the file part
                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if image_path[-4:] == '.mp4':
                        video_out = image_path[:-4] + '_detected' + image_path[-4:]
                        video_reader = cv2.VideoCapture(image_path)

                        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

                        video_writer = cv2.VideoWriter(video_out,
                                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                                       50.0,
                                                       (frame_w, frame_h))

                        for i in tqdm(range(nb_frames)):
                            _, image = video_reader.read()
                            time_now +=1

                            boxes = yolo.predict(image)
                            data_head = data_head.append({'Time': str(time_now//60)+'/'+str(time_now%60),'Head_count': len(boxes)}, ignore_index=True)
                            image = draw_boxes(image, boxes, config['model']['labels'])

                            video_writer.write(np.uint8(image))
                            if(time_now==20):
                                break
                        video_reader.release()
                        video_writer.release()
                        data_head.to_csv('head_count.csv',index=False)

                        temp1 = filename
                        print('temp1', temp1)
                    return redirect(url_for('result'))

            elif request.form['btn'] == 'Testing':
                return redirect(url_for('testdrive'))
        else:
            return render_template('block1.html')

@app.route("/block2", methods=['GET', 'POST'])
def block2():
    if not session.get('logged_in'):
        flash('Please login to access this page', 'info')
        return redirect(url_for('login_b2'))
    else:
        global temp1
        if request.method == 'POST':

            if request.form['btn'] == 'Upload':
                # check if the post request has the file part
                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if image_path[-4:] == '.mp4':
                        video_out = image_path[:-4] + '_detected' + image_path[-4:]
                        video_reader = cv2.VideoCapture(image_path)

                        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

                        video_writer = cv2.VideoWriter(video_out,
                                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                                       50.0,
                                                       (frame_w, frame_h))

                        for i in tqdm(range(nb_frames)):
                            _, image = video_reader.read()

                            # boxes = yolo.predict(image)
                            # print(image)
                            # image = draw_boxes(image, boxes, config['model']['labels'])

                            video_writer.write(np.uint8(image))

                        video_reader.release()
                        video_writer.release()
                        temp1 = filename + '_detected' + image_path[-4:]
                        print('temp1', temp1)
                    return redirect(url_for('result'))

            elif request.form['btn'] == 'Testing':
                return redirect(url_for('testdrive2'))
        else:
            return render_template('block2.html')
import base64

# Vivasvan

@app.route("/testdrive", methods=['GET', 'POST'])
def testdrive():
    if request.method=="POST":
        data_url = request.form['someText']
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        # body = np.array(body)

        # Do thing here
        temp1 = ""
        return render_template('result.html',temp1 = UPLOAD_FOLDER+temp1)

        # return render_template('testdrive.html')
    else:
        return render_template('testdrive.html')


@app.route("/testdrive2", methods=['GET', 'POST'])
def testdrive2():
    if request.method=="POST":
        data_url = request.form['someText']
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        image = np.array(body)

        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])
        print(len(boxes), 'boxes are found')
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], "image")
        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        # Do thing here
        temp1 = "image" + '_detected' + image_path[-4:]
        return render_template('result.html',temp1 = UPLOAD_FOLDER+temp1)

        # return render_template('testdrive.html')
    else:
        return render_template('testdrive.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@cavity.com' and form.password.data == 'password':
            session['logged_in'] = True
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Incorrect entries,Please check email and password', 'danger')
            return render_template('login.html', title='Login', form=form)
    else:
        return render_template('login.html', title='Login', form=form)

@app.route("/login_b1", methods=['GET', 'POST'])
def login_b1():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@cavity.com' and form.password.data == 'password':
            session['logged_in'] = True
            flash('You have been logged in!', 'success')
            return redirect(url_for('block1'))
        else:
            flash('Incorrect entries,Please check email and password', 'danger')
            return render_template('login.html', title='Login', form=form)
    else:
        return render_template('login.html', title='Login', form=form)

@app.route("/login_b2", methods=['GET', 'POST'])
def login_b2():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@cavity.com' and form.password.data == 'password':
            session['logged_in'] = True
            flash('You have been logged in!', 'success')
            return redirect(url_for('block2'))
        else:
            flash('Incorrect entries,Please check email and password', 'danger')
            return render_template('login.html', title='Login', form=form)
    else:
        return render_template('login.html', title='Login', form=form)



@app.route("/logout")
def logout():
    session['logged_in'] = False
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))









