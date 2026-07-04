from flask import Flask, render_template, request
from recommender import recommend_phone

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    phones = recommend_phone(
        price=int(request.form["price"]),
        is5g=int(request.form["is5g"]),
        battery=int(request.form["battery"]),
        fast_charging=int(request.form["fast_charging"]),
        ram=int(request.form["ram"]),
        storage=int(request.form["storage"]),
        refresh_rate=int(request.form["refresh_rate"]),
        rear_camera=int(request.form["rear_camera"]),
        front_camera=int(request.form["front_camera"]),
        screen_size=float(request.form["screen_size"]),
        processor=request.form["processor"],
        os=request.form["os"]
    )

    phones = phones.to_dict(orient="records")

    return render_template(
        "index.html",
        phones=phones
    )


if __name__ == "__main__":
    app.run(debug=True)