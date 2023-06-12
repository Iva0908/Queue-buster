import os
from ultralytics import YOLO
import cv2

IMAGE_DIR = os.path.join('.', 'image')
image_path = os.path.join(IMAGE_DIR, 'download (1).jpeg')
image_path_out = os.path.join('static','output', 'output_image.jpg')

# Load the input image
frame = cv2.imread(image_path)
H, W, _ = frame.shape

# Create the output image with the same dimensions as the input image
output_frame = frame.copy()

model_path = os.path.join('.', 'runs', 'detect', 'train11', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.0

class_name_dict = {0: 'apple', 1: 'banana', 2: 'orange'}

results = model(frame)[0]
output=[]
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    output.append(float(result[5]))
    if score > threshold:
        cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(output_frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the annotated image
print(output)
cv2.imwrite(image_path_out, output_frame)

cv2.imshow('Annotated Image', output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

from app import app

if __name__ == '__main__':
    app.run(debug=True)
