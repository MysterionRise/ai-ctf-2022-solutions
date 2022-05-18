import tensorflow_hub as hub
import tensorflow as tf

progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
image = progan(tf.random.normal([1, 512]))
print(image)