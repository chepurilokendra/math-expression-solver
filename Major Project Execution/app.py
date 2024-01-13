from flask import Flask,render_template,request
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import base64
from final import read_image

#init flask app
app = Flask(__name__)
app.config['SECRET_KEY']="secretkey"
app.config['UPLOAD_FOLDER'] = "static"
CORS(app)
#flask server routes

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("File Upload")

@app.route("/",methods=["GET","POST"])
@app.route("/home",methods=["GET","POST"])
def home():
    # form = UploadFileForm()
    # if form.validate_on_submit():
        # file=form.file.data
        # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
    return render_template("index.html")

@app.route("/input",methods=["GET","POST"])
def input():
    output = request.form.to_dict()
    if output:
        # print(output)
        imgsrc = output["imgsrc"]
        # file1 = open("MyFile.txt", "w")
        # file1.write(imgsrc)
        imgsrc = imgsrc[imgsrc.index(',')+1:]

        # file = open('MyFile.txt', 'rb')
        # encoded_data = file.read()
        # file.close()
        #decode base64 string data
        decoded_data=base64.b64decode(imgsrc)
        #write the decoded data back to original format in  file
        img_file = open('input.jpg', 'wb')
        img_file.write(decoded_data)
        img_file.close()

        img_path = os.path.abspath("input.jpg")
        result = read_image(img_path)
        print("in flask ",result)
        equation,solution = result[0],result[1]


        # decoded_data=base64.b64decode((imgsrc))
    return render_template("index.html",equation=equation,result=solution)
    

# run server
if __name__ == "__main__":
    app.run(debug= True)