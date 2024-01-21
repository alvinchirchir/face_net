from PIL import Image
from numpy import asarray, expand_dims, linalg
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# Method to extract Face
def extract_face(image_path):
    img = Image.open(image_path).convert('RGB')
    pixels = asarray(img)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    if not faces:
        raise Exception("No face found in the provided image.")

    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)
    x2, y2 = abs(x + w), abs(y + h)
    face_array = pixels[y:y2, x:x2]
    


    image1 = Image.fromarray(face_array, 'RGB')
    image1 = image1.resize((224, 224))

    face_array = asarray(image1)

    confidence = faces[0]['confidence']

    return face_array, float(confidence)

# Generalize the data and extract embeddings
def extract_embeddings(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std = face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = tf.reduce_sum(embedding1 * embedding2)
    norm1 = tf.norm(embedding1)
    norm2 = tf.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity.numpy()

# Calculate Euclidean distance 
def euclidean_distance(embedding1, embedding2):
    distance = linalg.norm(embedding1 - embedding2)
    return (distance)

def verify_faces(image1, image2):
    model = tf.keras.applications.ResNet50(weights='imagenet')

    face1, confidence1 = extract_face(image1)

    face2, confidence2 = extract_face(image2)
    embedding1 = extract_embeddings(model, face1)
    embedding2 = extract_embeddings(model, face2)

    similarity_score = cosine_similarity(embedding1, embedding2)

    distance_score = euclidean_distance(embedding1, embedding2)
    print("done")

    # Define thresholds for similarity and distance
    similarity_threshold = 0.85
    distance_threshold = 1.0

    # Check if the faces are similar based on the thresholds
    if similarity_score > similarity_threshold and distance_score < distance_threshold:
        return {"result": "Faces are similar", "similarity_score": float(similarity_score), "distance_score": float(distance_score)}
    else:
        return  {"result": "Faces are not similar", "similarity_score": float(similarity_score), "distance_score":float( distance_score)}



