
import os
from django.shortcuts import render
import requests
import pytz
# import ipdb

from flask import Flask, render_template,redirect,request,session,flash
from flask_session import Session
# from helpers import apology, login_required, lookup, usd
from flask_login import login_required,LoginManager
from werkzeug.security import check_password_hash, generate_password_hash
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
from werkzeug import utils

app=Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Custom filter
# app.jinja_env.filters["usd"] = usd

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app) 

# Configure session to use filesystem (instead of signed cookies)
# app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

engine=create_engine('mysql+pymysql://root:Rkkr@26109@localhost/plantdiseasedetection')
db = scoped_session(sessionmaker(bind=engine)) 


model= keras.models.load_model("./models/mobileNet_V3_fineTuning.h5")

@app.route("/")
def index():
    return render_template ("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():


    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            flash('Username cannot be empty!')
            return render_template("login.html")

        # Ensure password was submitted
        elif not request.form.get("password"):
            flash('Password cannot be empty!')
            return render_template("login.html")

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = :username",
                          {"username":request.form.get("username")}).fetchall()

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            flash("Invalid username or password!")
            return render_template("login.html")

        # Remember which user has logged in
        session["user"] = rows[0]["username"]

        # Redirect user to home page
        return redirect("/check")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    else:
        username = request.form.get("username")
        users=db.execute("SELECT username FROM users").fetchall()
        for user in users:
            if user['username'] == username:
                flash('Username already exist!')
                return render_template('register.html')

        hash = request.form.get("password")
        conhash = request.form.get("confirmation")
        email = request.form.get("email")
        if hash == conhash:
            hash = generate_password_hash(hash)
            db.execute("INSERT INTO users (username, hash, email) VALUES (:username, :hash, :email)", {"username":username, "hash":hash, "email":email})
            db.commit()
            return redirect("/login")
                
        else:
            flash('Passwords does not match!')
            return render_template('register.html')


@app.route("/check")
# @login_required
def check():
    return render_template('home.html')


@app.route("/logout")
# @login_required
def logout():
    # Forget any user_id
    session.clear()
    # Redirect user to login form
    return redirect("/login")


@app.route("/upload", methods=["POST"])
def upload():
    global secure_filename
    image_file=request.files["filename"]
    secure_filename=utils.secure_filename(image_file.filename)
    image_path=os.path.join(app.root_path, secure_filename)
    image_file.save(image_path)
    return redirect("/predict")
    # else:
    #     flash("Image upload failed!")
    #     return render_template("home.html")


@app.route("/predict")
def predict():
    global secure_filename
    try:
        img=Image.open(os.path.join(app.root_path,secure_filename))
        img=img.resize((224,224))
        img=np.asarray(img)
        img=np.reshape(img,(1,224,224,3))
        # ipdb.set_trace()
        out=model.predict(img)
        plant_disease_labels=['Apple_Apple scab',
            'Apple_Black rot',
            'Apple_Cedar apple rust',
            'Apple_Healthy',
            'Blueberry_Healthy',
            'Cherry_Powdery mildew',
            'Cherry_Healthy',
            'Corn_Cercospora gray leaf spot',
            'Corn_Common rust',
            'Corn_Northern leaf blight',
            'Corn_Healthy',
            'Grape_Black rot',
            'Grape_Esca black measles',
            'Grape_Isariopsis leaf spot',
            'Grape_Healthy',
            'Orange_Haunglongbing citrus greening',
            'Peach_Bacterial spot',
            'Peach_Healthy',
            'Pepper Bell_Bacterial spot',
            'Pepper Bell_Healthy',
            'Potato_Early blight',
            'Potato_Late blight',
            'Potato_Healthy',
            'Raspberry_Healthy',
            'Soybean_Healthy',
            'Squash_Powdery mildew',
            'Strawberry_Leaf scorch',
            'Strawberry_Healthy',
            'Tomato_Bacterial spot',
            'Tomato_Early blight',
            'Tomato_Late blight',
            'Tomato_Leaf mold',
            'Tomato_Septoria leaf spot',
            'Tomato_Spider mites two spotted spider mite',
            'Tomato_target spot',
            'Tomato_Tomato yellow leaf curl virus',
            'Tomato_Tomato mosaic virus',
            'Tomato_Healthy'
            ]
        predicted_class= np.argmax(out)
        print(predicted_class)
        predicted_class= plant_disease_labels[predicted_class]
        print(predicted_class)
        predicted_crop=predicted_class.split("_")
        print(predicted_crop)
        return render_template("predict.html",predicted_crop=predicted_crop[0], predicted_disease=predicted_crop[1])
    except Exception as e:
        return str(e)