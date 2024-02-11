from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Directory where images are stored
IMAGE_FOLDER = "C:/Users/kris_/Python/tensor-sd/TensorRT/demo/Diffusion\output"

# Global variable to keep track of the current latest image
current_latest_image = None

@app.route('/')
def index():
    global current_latest_image
    # Find the latest image file in the IMAGE_FOLDER
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    latest_image_file = max(image_files, key=os.path.getmtime) if image_files else None
    latest_image_name = os.path.basename(latest_image_file) if latest_image_file else None

    # If there is a new latest image, delete the previous one
    if latest_image_name and latest_image_name != current_latest_image:
        if current_latest_image:
            try:
                os.remove(os.path.join(IMAGE_FOLDER, current_latest_image))
            except OSError as e:
                print(f"Error deleting file {current_latest_image}: {e.strerror}")
        current_latest_image = latest_image_name

    return render_template('index2.html', latest_image_name=latest_image_name)

@app.route('/images/<filename>')
def image(filename):
    # Serve the image file
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)