from flask import Flask, render_template, request
from predict import predict_waste
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(path)

    result = predict_waste(path)

    return render_template(
        "result.html",
        prediction=result,
        image=image.filename
    )


if __name__ == "__main__":
    app.run(debug=True)