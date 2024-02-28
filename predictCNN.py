import pickle
import librosa
import numpy as np
from tensorflow.keras.models import load_model

def predict_emotion_from_file(file_path, model, label_encoder):
    data, sampling_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40)
    if mfccs.shape[1] > 180:
        mfccs = mfccs[:, :180]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, 180 - mfccs.shape[1])), "constant")
    mfccs = np.expand_dims(mfccs, axis=-1)
    mfccs = np.expand_dims(mfccs, axis=0) 
    prediction = model.predict(mfccs)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_emotion = label_encoder.inverse_transform(predicted_index)
    return predicted_emotion[0]

model = load_model('cnn.h5')  

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

predicted_emotion = predict_emotion_from_file('Snimka (2).wav', model, label_encoder)
print(f"Predviđena emocija: {predicted_emotion}")