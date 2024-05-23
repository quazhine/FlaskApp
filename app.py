""" ฝั่งของเว็บ """
from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf

""" ฝั่งของการดึงโมเดลมาใช้งาน """
import librosa
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


#สร้างโฟลเดอร์อัพโหลด ถ้าไม่มีก็สร้างขึ้นมา
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'wav','mp3'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#โหลดโมเดล&ปรับไฟล์
model = load_model('best_model_DNN.h5')
scaler = joblib.load('Model_DNN_scaler_.pkl')



#หน้าหลัก เตรียมเปลี่ยนเป็นอัพโหลด
@app.route('/')
def index():
    return render_template('upload.html') 

""" #หน้าอัพโหลด
@app.route('/upload')
def uploadpage():
    return render_template('upload.html') """

#ฟังก์ชั่นของไฮน์1
def extract_features(y, sr, n_mfcc=40):
    try:
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        harm, perc = librosa.effects.hpss(y)

        features = {
            'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rms_mean': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'spectral_rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'spectral_contrast_mean': np.mean(contrast),
            'harmonic_mean': np.mean(harm),
            'percussive_mean': np.mean(perc),
            'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y=y))
        }
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc)
        feature_values = np.array(list(features.values())).reshape(1, -1)
        return feature_values
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((1, 49))  # Return a row of zeros if there is an error


#ฟังก์ชั่นของไฮน์2
def predict_song_genre(file_path, model, scaler, segment_duration=30):
    y, sr = librosa.load(file_path, sr=None)  # Load the whole song
    song_duration = librosa.get_duration(y=y, sr=sr)
    predictions = []
    
    for start in np.arange(0, song_duration, segment_duration):
        end = start + segment_duration
        if end > song_duration:
            end = song_duration
        y_segment = y[int(start * sr):int(end * sr)]
        features = extract_features(y_segment, sr)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_class_index = np.argmax(prediction)
        predictions.append(predicted_class_index)
        
    # Analyze the results for all segments
    from collections import Counter
    most_common = Counter(predictions).most_common(1)
    print(f"Most predicted class for {file_path} is: {most_common[0][0]} with {most_common[0][1]} out of {len(predictions)} segments")
    return predictions


#อัพโหลด
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3')):
        filename = file.filename
        #1
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      #  print('1'+filename)
        #file_path = 'uploads/'
        #file_path = 'uploads/{filename}'
      #  file_path = 'uploads/{}'.format(filename)
        
        #2
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)


        result = predict_song_genre(file_path, model, scaler)
        print('before delete'+file_path)

        #ลบไฟล์ที่อัพโหลดมา หลังพรีดิก
        os.remove(file_path)

        return redirect(url_for('result', result=result))
    else:
        return "Invalid file type. Please upload a .wav or .mp3 file"


"""
file_path = 'uploads/'
predict_song_genre(file_path, model, scaler)
"""


"""
0blues
1class
2disco
3hiphop
4metal
5pop
6rock
"""
@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))




