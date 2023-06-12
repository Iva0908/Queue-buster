from flask import Flask, render_template, request, jsonify, redirect
import os
from ultralytics import YOLO
import cv2
import sys
from io import StringIO
app = Flask(__name__)


# def start_capture():
#     capture_image()
#     return 'Image captured successfully!'
@app.route('/capture')
def capture_image():
    check=True

    while(check):
        check = False
        camera = cv2.VideoCapture(0)  # Open the camera (0 for the default camera)

        while True:

            ret, frame = camera.read()  # Read a frame from the camera

            cv2.imshow('Camera', frame)  # Display the frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
                image_path = os.path.join('static','uploads','captured_image.jpg')
                cv2.imwrite(image_path, frame)  # Save the image as 'captured_image.jpg'
                break

        camera.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all windows

        image_path = os.path.join('static','uploads','captured_image.jpg')


        # Load the input image
        frame = cv2.imread(image_path)
        H, W, _ = frame.shape

        # Create the output image with the same dimensions as the input image
        output_frame = frame.copy()

        model_path = os.path.join('.', 'runs', 'detect', 'train11', 'weights', 'last.pt')

        # Capture the model output
        output_buffer = StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer

        # Load a model
        model = YOLO(model_path)  # load a custom model

        threshold = 0.3

        class_name_dict = {0: 'apple', 1: 'banana', 2: 'orange'}

        results = model(frame)[0]
        output = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            output.append(int(result[5]))
            if score > threshold:
                cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(output_frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Restore the standard output and retrieve the model output
        sys.stdout = old_stdout
        model_output = output_buffer.getvalue()

        # Save the annotated image
        output_image_path = os.path.join('static', 'output', 'output_image.jpg')
        cv2.imwrite(output_image_path, output_frame)
        image1 = os.path.join('static', 'images', "image1.jpeg")
        image2 = os.path.join('static', 'images', "image2.jpeg")
        image3 = os.path.join('static', 'images', "image3.jpeg")
        print(image_path)
        response = {
        'image1_path': image_path,
        'image2_path': output_image_path,
        'console_output': output
        }

        return render_template('index.html', image_path=image_path, output_image_path=output_image_path,console_output=output, image1_path=image1, image2_path=image2, image3_path=image3)





# ...

@app.route('/')
def hello():
    image1 = os.path.join('static','images',"image1.avif")
    image2 = os.path.join('static','images',"image1.jpeg")
    return render_template('index.html', image_path=image2,output_image_path=image1)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is selected
        if 'image' not in request.files:
            return render_template('index.html', error='No image selected', console_output='')

        image_file = request.files['image']

        # Save the uploaded image
        image_path = os.path.join('static', 'uploads', image_file.filename)
        image_file.save(image_path)

        # Load the input image
        frame = cv2.imread(image_path)
        H, W, _ = frame.shape

        # Create the output image with the same dimensions as the input image
        output_frame = frame.copy()

        model_path = os.path.join('.', 'runs', 'detect', 'train11', 'weights', 'last.pt')

        # Capture the model output
        output_buffer = StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer

        # Load a model
        model = YOLO(model_path)  # load a custom model

        threshold = 0.3

        class_name_dict = {0: 'apple', 1: 'banana', 2: 'orange'}

        results = model(frame)[0]
        output = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            output.append(int(result[5]))
            if score > threshold:
                cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(output_frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Restore the standard output and retrieve the model output
        sys.stdout = old_stdout
        model_output = output_buffer.getvalue()


        # Save the annotated image
        output_image_path = os.path.join('static', 'output', 'output_image.jpg')
        cv2.imwrite(output_image_path, output_frame)

        return render_template('index.html', image_path=image_path, output_image_path=output_image_path,
                               console_output=output)

    return render_template('index.html', console_output='')


if __name__ == '__main__':
    app.run(debug=True)

