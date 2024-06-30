import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import keras

app = Flask(__name__)

model = keras.models.load_model('eyee.keras')  # Ensure this path is correct

# Define class labels
class_labels = ['Uveitis', 'Normal_Eyee']

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/eye', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            img = Image.open(file)
            img_array = preprocess_image(img)
            
            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            class_probabilities = predictions[0].tolist()

            return jsonify({
                'predicted_class': predicted_class_label,
                'class_probabilities': dict(zip(class_labels, class_probabilities))
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
