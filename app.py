from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import script as model

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")

@app.route('/predict', methods = ['GET', 'POST'])
def cek():
  if request.method == 'POST':
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return model.predict(f.filename)

if __name__ == '__main__':
	app.run(debug=True)