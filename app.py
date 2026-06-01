from flask import Flask, render_template, Response, jsonify, request
import cv2
from deepface import DeepFace

app = Flask(__name__)

camera = cv2.VideoCapture(0)

latest_emotion = "neutral"


def analyze_text(text):

    text = text.lower().strip()

    lie_words = [
        "i am god",
        "i own mars",
        "i can fly",
        "i am immortal"
    ]

    suspicious_words = [
        "kill",
        "killed",
        "murder",
        "stole",
        "crime",
        "robbed",
        "hack",
        "hacked",
        "maybe",
        "perhaps"
    ]

    for w in lie_words:
        if w in text:
            return {
                "label": "LIE DETECTED",
                "score": 95,
                "color": "red"
            }

    for w in suspicious_words:
        if w in text:
            return {
                "label": "SUSPICIOUS",
                "score": 75,
                "color": "orange"
            }

    return {
        "label": "TRUTH",
        "score": 90,
        "color": "lime"
    }


def gen():
    global latest_emotion

    while True:
        ok, frame = camera.read()

        if not ok:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            latest_emotion = result[0]['dominant_emotion']

            faces = DeepFace.extract_faces(frame, enforce_detection=False)

            for f in faces:

                area = f["facial_area"]

                x = area["x"]
                y = area["y"]
                w = area["w"]
                h = area["h"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                l = 20

                cv2.line(frame, (x, y), (x + l, y), (0, 255, 0), 2)
                cv2.line(frame, (x, y), (x, y + l), (0, 255, 0), 2)

                cv2.line(frame, (x + w, y), (x + w - l, y), (0, 255, 0), 2)
                cv2.line(frame, (x + w, y), (x + w, y + l), (0, 255, 0), 2)

                cv2.line(frame, (x, y + h), (x + l, y + h), (0, 255, 0), 2)
                cv2.line(frame, (x, y + h), (x, y + h - l), (0, 255, 0), 2)

                cv2.line(frame, (x + w, y + h), (x + w - l, y + h), (0, 255, 0), 2)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - l), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    latest_emotion.upper(),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        except:
            pass

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/emotion")
def emotion():
    return render_template("emotion.html")


@app.route("/lie")
def lie():
    return render_template("lie.html")


@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/emotion_data")
def emo_data():

    status_map = {
        "happy": "Positive Mood",
        "sad": "Low Mood",
        "angry": "Aggressive Mood",
        "fear": "Fear Detected",
        "surprise": "Surprised",
        "neutral": "Neutral State"
    }

    analysis_map = {
        "happy": "User appears happy and relaxed.",
        "sad": "User appears emotionally low.",
        "angry": "User appears frustrated.",
        "fear": "User appears nervous.",
        "surprise": "Unexpected reaction detected.",
        "neutral": "No strong emotion detected."
    }

    return jsonify({
        "emotion": latest_emotion,
        "status": status_map.get(latest_emotion, "Unknown"),
        "analysis": analysis_map.get(latest_emotion, "No analysis")
    })


@app.route("/lie_text", methods=["POST"])
def lie_text():
    data = request.json
    return jsonify(analyze_text(data["text"]))


if __name__ == "__main__":
    app.run(debug=True)