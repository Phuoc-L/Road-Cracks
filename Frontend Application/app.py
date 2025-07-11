from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'

# Ensure content folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO("models/Custom_Model2_openvino_model", task="detect")
# model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded video
        file = request.files['video']
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Process the video
        output_filename = "output_video.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, verbose=False, stream=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return render_template('index.html', video_url=output_filename)

    return render_template('index.html', video_url=None)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
