import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageFilter
from tqdm import tqdm


def update_images():
    # #Create fake face for web page
    progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
    path = 'static/images_base/'
    for background in os.listdir(path):
        base_im = Image.open(path + background).convert('RGBA')
        # base_im.putalpha(1)
        for x in tqdm(range(0, base_im.size[0], 128)):
            for y in range(0, base_im.size[1], 128):
                img = progan(tf.random.normal([512]))['default'][0]
                im = Image.fromarray(np.array(img * 255).astype(np.uint8))
                # im.putalpha(128)
                base_im.paste(im, (x, y))

        blurImage = base_im.filter(ImageFilter.GaussianBlur(2.5))
        mix_img = Image.blend(Image.open(path + background).convert('RGBA'), blurImage, alpha=0.35)
        mix_img = mix_img.convert("RGB")
        mix_img.save('static/images/' + background)


if __name__ == '__main__':
    update_images()
