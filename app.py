from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3

app = Flask(__name__)

# Load AI model
model = joblib.load("crop_model.pkl")

# Database setup
def init_db():
    conn = sqlite3.connect("farmers.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY,
            rainfall REAL,
            temperature REAL,
            fertilizer REAL,
            prediction REAL
        )
    """)

    conn.commit()
    conn.close()

init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["rainfall"])
    temp = float(request.form["temperature"])
    fert = float(request.form["fertilizer"])

    prediction = model.predict([[rainfall, temp, fert]])[0]

    # Save to DB
    conn = sqlite3.connect("farmers.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO records (rainfall, temperature, fertilizer, prediction)
        VALUES (?, ?, ?, ?)
    """, (rainfall, temp, fert, prediction))

    conn.commit()
    conn.close()

    return jsonify({
        "yield": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
