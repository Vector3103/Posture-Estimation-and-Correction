from flask import Flask, render_template, Response
import cv2
import time
import pickle as pk
import mediapipe as mp
import pandas as pd
import pyttsx4
import multiprocessing as mtp

def init_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 360)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cam.set(cv2.CAP_PROP_SHARPNESS, 125)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def get_pose_name(index):
    names = {
        0: "Adho Mukha Svanasana",
        1: "Phalakasana",
        2: "Utkata Konasana",
        3: "Vrikshasana",
    }
    return str(names[index])


def init_dicts():
    landmarks_points = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }
    landmarks_points_array = {
        "left_shoulder": [], "right_shoulder": [],
        "left_elbow": [], "right_elbow": [],
        "left_wrist": [], "right_wrist": [],
        "left_hip": [], "right_hip": [],
        "left_knee": [], "right_knee": [],
        "left_ankle": [], "right_ankle": [],
        "left_heel": [], "right_heel": [],
        "left_foot_index": [], "right_foot_index": [],
    }
    col_names = []
    for i in range(len(landmarks_points.keys())):
        name = list(landmarks_points.keys())[i]
        col_names.append(name + "_x")
        col_names.append(name + "_y")
        col_names.append(name + "_z")
        col_names.append(name + "_v")
    cols = col_names.copy()
    return cols, landmarks_points_array

def cv2_put_text(image, message):
    cv2.putText(
        image,
        message,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        5,
        cv2.LINE_AA
    )


def destory(cam, tts_proc, tts_q):
    cv2.destroyAllWindows()
    cam.release()
    tts_q.put(None)
    tts_q.close()
    tts_q.join_thread()
    tts_proc.join()


# Import the functions and constants from the provided code
from landmarks import extract_landmarks
from recommendations import check_pose_angle
from calc_angles import rangles

app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx4.init()
tts_last_exec = time.time()

# Initialize multiprocessing queue for text-to-speech
tts_q = mtp.JoinableQueue()

def tts_process(tts_q):
    engine = pyttsx4.init()
    while True:
        text = tts_q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

tts_proc = mtp.Process(target=tts_process, args=(tts_q,))
tts_proc.start()

def generate_frames():
    cam = cv2.VideoCapture(0)
    model = pk.load(open("./models/4_poses.model", "rb"))
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cols = init_dicts()
    landmarks_points_array = init_dicts()
    angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")

    while True:
        success, image = cam.read()
        if not success:
            break
        flipped = cv2.flip(image, 1)
        resized_image = cv2.resize(flipped, (640, 360), interpolation=cv2.INTER_AREA)

        err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)

        if err == False:
            prediction = model.predict(df)
            probabilities = model.predict_proba(df)

            mp_drawing.draw_landmarks(flipped, landmarks, mp_pose.POSE_CONNECTIONS)

            if probabilities[0, prediction[0]] > 0.85:
                cv2.putText(flipped, get_pose_name(prediction[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                angles = rangles(df, landmarks_points_array)
                suggestions = check_pose_angle(prediction[0], angles, angles_df)

                if time.time() > tts_last_exec:
                    tts_q.put(suggestions[0])
                    tts_last_exec = time.time() + 5

            else:
                cv2.putText(flipped, "No Pose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', flipped)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
