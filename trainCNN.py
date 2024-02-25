import os
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(dataset_path):
    features, emotions = [], []
    for file in os.listdir(dataset_path):
        parts = file.split("-")
        if len(parts) >= 3:
            try:
                emotion_code = int(parts[2])
                emotion = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}.get(emotion_code, None)
                if emotion is not None:
                    path = os.path.join(dataset_path, file)
                    data, sampling_rate = librosa.load(path)
                    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40)
                    if mfccs.shape[1] > 180:
                        mfccs = mfccs[:, :180]
                    else:
                        mfccs = np.pad(mfccs, ((0, 0), (0, 180 - mfccs.shape[1])), "constant")
                    features.append(mfccs)
                    emotions.append(emotion)
            except ValueError:
                print(f"Skipping file with unexpected format: {file}")
    return np.array(features), np.array(emotions)

features, emotions = load_data('data')

label_encoder = LabelEncoder()
emotions_encoded = label_encoder.fit_transform(emotions)
emotions_one_hot = to_categorical(emotions_encoded)

features = np.expand_dims(features, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(features, emotions_one_hot, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, 180, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(emotions_encoded)), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
model.save('cnn.h5')  
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

