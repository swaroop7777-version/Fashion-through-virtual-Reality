import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import os
vgg_model= VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def load_fashion_features(dataset_url):
    dataset_path = 'fashion dataset.zip'
    if not os.path.exists(dataset_path):
        r = requests.get(dataset_url)
        with open(dataset_path, 'wb') as f:
            f.write(r.content)
    return pd.read_csv('features.csv')
fashion_data = load_fashion_features('DATASET_LINK_HERE')
def extract_features(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    return features.flatten()

def recommend_clothing(face_img):
    face_features = extract_features(face_img)
    similarities = []
    for _, row in fashion_data.iterrows():
        cloth_features = np.fromstring(row['features'][1:-1], sep=' ')
        sim = cosine_similarity([face_features], [cloth_features])[0][0]
        similarities.append((row['image_path'], sim))
    best_match = sorted(similarities, key=lambda x: x[1], reverse=True)[0]
    return best_match[0]
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        cloth_path = recommend_clothing(face_img)
        cv2.putText(frame, f"Recommended: {cloth_path}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Fashion Recommender', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
