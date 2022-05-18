import torch
from PIL import Image
import numpy as np

x = torch.load("backup/data.pt")


for i in range(6):
    np.save(f"face{i}", x[0][i])

# img = Image.fromarray(x[0][0].numpy())
#
# img.save("test.png")