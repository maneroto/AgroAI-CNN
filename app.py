import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from predict import get_predictions, get_classes

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make predictions
        _, top_index = get_predictions(file_path)
        classes_df = get_classes()
        class_name = classes_df.loc[classes_df['class_index'] == top_index, 'class'].values[0]

        return str(class_name)
    return None


if __name__ == '__main__':
    app.run(debug=True)