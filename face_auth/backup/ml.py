import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets


#Creating face embeding vectors from dir
#Will save in data.pt
def face_to_vec():
    mtcnn = MTCNN(image_size=150)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    dataset = datasets.ImageFolder('valid_user/')
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    def collate_fn(x):
        return x[0]
    loader = DataLoader(dataset, collate_fn=collate_fn)

    #create vector form face
    name_list = []
    embedding_list = []

    for img, idx in loader:
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt')
    print("Database face to vec successfully created!")


if __name__ == "__main__":
    face_to_vec()
