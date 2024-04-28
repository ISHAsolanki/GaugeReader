import os
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import threading
from playsound import playsound

# Flask app setup
app = Flask(__name__)

# Gauge parameters
min_angle = 0.0  # Minimum angle on the gauge dial in degrees
max_angle = 360.0  # Maximum angle on the gauge dial in degrees
min_value = 0.0  # Minimum reading value (like 0 psi)
max_value = 200.0  # Maximum reading value (like 200 psi)
units = 'psi'  # Units for the gauge

# Threshold for triggering the beep sound
threshold_limit = 100.0  # Default threshold (in psi)
current_reading = None  # Variable to store the current reading


# Function to play a beep sound
def play_beep():
    playsound("beep.mp3")  


# Define a function to calculate the distance between two points
def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate the gauge reading from the detected line (needle)
def calculate_gauge_reading(img, min_angle, max_angle, min_value, max_value):
    gray = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 3), cv2.COLOR_BGR2GRAY)

    # Detect circles to find the gauge face
    height, width = img.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=50,
        minRadius=int(height * 0.35),
        maxRadius=int(height * 0.48),
    )

    if circles is None:
        return None, None

    # Find the circle with the closest average center
    circles = np.round(circles[0, :]).astype("int")
    avg_circle = circles[0]
    avg_x, avg_y, avg_r = avg_circle[0], avg_circle[1], avg_circle[2]

    # Threshold for line detection
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        image=thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=15,
        maxLineGap=5,
    )

    if lines is None:
        return None, None

    # Find the line that best represents the gauge needle
    best_line = None
    best_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1 = dist_2_pts(avg_x, avg_y, x1, y1)
        dist2 = dist_2_pts(avg_x, avg_y, x2, y2)

        if (0.3 * avg_r <= dist1 <= 0.7 * avg_r) or (0.3 * avg_r <= dist2 <= 0.7 * avg_r):
            length = dist_2_pts(x1, y1, x2, y2)  # Line segment length
            if length > best_length:
                best_length = length
                best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    # Determine which point is the needle's tip
    x1, y1, x2, y2 = best_line
    dist1 = dist_2_pts(avg_x, avg_y, x1, y1)
    dist2 = dist_2_pts(avg_x, avg_y, x2, y2)

    needle_tip = (x1, y1) if dist1 > dist2 else (x2, y2)

    # Calculate the angle from the gauge's center to the needle's tip
    angle_rad = np.arctan2(needle_tip[1] - avg_y, needle_tip[0] - avg_x)
    angle_deg = np.degrees(angle_rad) + 90

    if angle_deg < 0:
        angle_deg += 360

    old_range = max_angle - min_angle
    new_range = max_value - min_value
    reading = ((angle_deg - min_angle) * new_range) / old_range + min_value

    return best_line, reading


# Function to generate frames from the webcam feed
def generate_frames():
    global current_reading

    cap = cv2.VideoCapture(0)  # Open default camera (usually 0)
    if not cap.isOpened():
        raise Exception("Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        best_line, reading = calculate_gauge_reading(frame, min_angle, max_angle, min_value, max_value)

        if best_line:
            x1, y1, x2, y2 = best_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Reading: {reading:.2f} {units}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            current_reading = reading  # Store the latest gauge reading

            # Check if the reading exceeds the threshold
            if reading > threshold_limit:
                # Play the beep sound in a separate thread to avoid blocking
                threading.Thread(target=play_beep).start()

        else:
            cv2.putText(
                frame,
                "Gauge not detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the HTTP response stream
        yield (
            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


# Define the Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page


# Define the route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to get the current gauge reading
@app.route('/current_reading', methods=['GET'])
def get_current_reading():
    return jsonify({"reading": current_reading if current_reading is not None else "N/A"})


# Route to set the threshold limit for the gauge reading
@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global threshold_limit
    new_limit = float(request.form["threshold"])
    threshold_limit = new_limit
    return redirect(url_for("index"))  # Redirect back to the main page


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
