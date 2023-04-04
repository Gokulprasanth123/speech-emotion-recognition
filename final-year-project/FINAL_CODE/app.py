
from flask import Flask,render_template,request
import pickle
from werkzeug.utils import secure_filename
import numpy as np
import os
from model import extract_feature
# import librosa
# import soundfile
# from model import extract_feature
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
picFolder=os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picFolder
@app.route('/')
def hello_world():
    pic1=os.path.join(app.config['UPLOAD_FOLDER'],'images.png')
    return render_template('index.html',image=pic1)


@app.route('/predict',methods=['POST','GET'])
def predict():
   pic2=os.path.join(app.config['UPLOAD_FOLDER'],'happy.jfif')
   pic3=os.path.join(app.config['UPLOAD_FOLDER'],'sad.jfif')
   pic4=os.path.join(app.config['UPLOAD_FOLDER'],'fear.jfif')
   pic5=os.path.join(app.config['UPLOAD_FOLDER'],'angry.webp')
   pic6=os.path.join(app.config['UPLOAD_FOLDER'],'disgust.jfif')
   pic7=os.path.join(app.config['UPLOAD_FOLDER'],'calm.jfif')
   pic8=os.path.join(app.config['UPLOAD_FOLDER'],'background.webp')
   file=request.files['file']
   file.save(secure_filename(file.filename))
   res=extract_feature(secure_filename(file.filename),mfcc=True, chroma=True, mel=True)
   res=res.reshape(1,-1)
   result=model.predict(res)
   print(result)
   return render_template('predict.html',pred=result,happy=pic2,sad=pic3,fear=pic4,angry=pic5,disgust=pic6,calm=pic7,bg=pic8)


if __name__=='__main__':
    app.debug=True
    app.run(debug=False,host='0.0.0.0')