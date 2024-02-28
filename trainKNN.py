import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(dataset_path):
    features, emotions = [], []
    for file in os.listdir(dataset_path):
        emotion_code = int(file.split("-")[2])
        emotion = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}.get(emotion_code, None)
        if emotion is not None:
            path = os.path.join(dataset_path, file)
            data, sampling_rate = librosa.load(path)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
            emotions.append(emotion)
    return np.array(features), np.array(emotions)

features, emotions = load_data('data')

label_encoder = LabelEncoder()
emotions_encoded = label_encoder.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(features, emotions_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)

knn_history = knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Preciznost k-NN klasifikatora pri k={k}: {accuracy:.2f}")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrica konfuzije')
plt.xlabel('Predviđene klase')
plt.ylabel('Stvarne klase')
plt.show()

def extract_features(audio_path):
    data, sampling_rate = librosa.load(audio_path)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return mfccs

def predict_emotion(audio_path, model, scaler, label_encoder):
    features = extract_features(audio_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    predicted_emotion = label_encoder.inverse_transform(prediction)
    return predicted_emotion[0]

new_audio_path = '/path/to/new/audio/file.wav'
predicted_emotion = predict_emotion(new_audio_path, knn, scaler, label_encoder)
print(f"Prepoznana emocija iz glasa: {predicted_emotion}")