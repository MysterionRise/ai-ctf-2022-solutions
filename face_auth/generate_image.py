import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

def face_to_vec(img):
    # img = Image.open(image_path)
    mtcnn = MTCNN(image_size=150)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    face = mtcnn(img)
    emb_ = None
    if face is not None:
        emb_ = resnet(face.unsqueeze(0))
    return emb_


def display_image(image):
    image = tf.constant(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return PIL.Image.fromarray(image.numpy())


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = tf.norm(v1)
    v2_norm = tf.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = tf.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return vectors


progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

initial_vector = tf.random.normal([1, 512])
faces = torch.load("data.pt")[0]
# user = faces[0]
vectors = []
for i in range(1000):
    vectors.append(tf.random.normal([1, 512]))

dist = 100
prev_min_dist = 1.3
distances = []
images = []
best_img = None
while dist > 0.9:
    distances = []
    new_vectors = []
    for vector in vectors:
        arr = progan(vector)['default'][0]
        img = display_image(arr)
        emb = face_to_vec(img)
        if emb is not None:
            for user in faces:
                dist = torch.dist(user, emb).item()
                if dist < 0.9:
                    print(dist)
                    img.save("res.png")
                    best_img.save("best.png")
                    break
                if dist < prev_min_dist:
                    images.append(img)
                    # best_img = img
                    distances.append(dist)
                    new_vectors.extend(interpolate_hypersphere(user, vector, 50))
    if len(distances) > 0:
        prev_min_dist = min(distances)
        best_img = images[distances.index(prev_min_dist)]
        print(prev_min_dist)
    vectors = new_vectors
    if len(vectors) == 0:
        print(prev_min_dist)
        img.save("res.png")
        best_img.save("best.png")
        break