from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import script as model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'

@app.route("/")
def index():
  return "Hello world!"

@app.route('/predict', methods = ['GET', 'POST'])
def cek():
  if request.method == 'POST':
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return model.predict(f.filename)

if __name__ == '__main__':
	app.run(debug=True)
  