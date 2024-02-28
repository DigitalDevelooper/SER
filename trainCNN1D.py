import os
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,Conv1D, MaxPooling1D, Flatten, Dense, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(dataset_path, max_length=236):
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
                    mfccs = mfccs.T  
                    
                    if len(mfccs) > max_length:
                        mfccs = mfccs[:max_length]
                    else:
                        mfccs = np.pad(mfccs, ((0, max_length - len(mfccs)), (0, 0)), mode='constant', constant_values=0)
                    
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
    Conv1D(256, 5, padding='same', input_shape=(236, 40)),
    Activation('relu'),
    MaxPooling1D(pool_size=8),
    Dropout(0.2),
    Conv1D(128, 5, padding='same'),
    Activation('relu'),
    MaxPooling1D(pool_size=4),
    Dropout(0.1),
    Flatten(),
    Dense(64),
    Dense(8),  
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
model.save('cnn.h5')  
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Klasifikacijski izvještaj:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))


conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrica konfuzije')
plt.ylabel('Stvarne klase')
plt.xlabel('Predviđene klase')
plt.show()


plt.figure(figsize=(10, 8))
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
