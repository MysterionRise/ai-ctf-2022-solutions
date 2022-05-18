import os
import secrets
from zipfile import ZipFile

import torch
from PIL import Image
# ML libs
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, render_template, request, flash, send_file
from werkzeug.utils import secure_filename

from fake_face import update_images
from ml import face_to_vec

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
PASS_TRASHOLD = 0.9

app = Flask(__name__, template_folder='templates')
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # maximum file size

if not os.path.isfile("data.pt"):
    print("Database face_to_vec not found, Try to creating... ")
    face_to_vec()

# If we have no images - generate it
if len(os.listdir("static/images")) == 0:
    print("Creating face img")
    update_images()

#GET KEY
try:
    KEY = os.environ['KEY']
except KeyError:
    KEY = "Key not selected!"


# face recognition network
mtcnn = MTCNN(image_size=150)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
saved_data = torch.load('data.pt')
embedding_list = saved_data[0]
name_list = saved_data[1]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def start():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return render_template('validate.html', stat_auth="No file chosen ", key="please chose file with your face")

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            # CHECK FACE
            img = Image.open(file_path).convert('RGB')
            face = mtcnn(img)
            # check empty face
            if face == None:
                stat_auth = "can't find face"
                key = "Try again"
                return render_template('validate.html', stat_auth=stat_auth, key=key)
            # extract vector from face
            emb = resnet(face.unsqueeze(0)).detach()

            dist_list = []
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)
            idx_min = dist_list.index(min(dist_list))

            if dist_list[idx_min] <= PASS_TRASHOLD:

                stat_auth = f"access granted - you are {name_list[idx_min]}"
                key = KEY
                print(dist_list)
                return render_template('validate.html', stat_auth=stat_auth, key=key)
            else:
                stat_auth = "access denied"
                key = "try again"
                print(dist_list)
                return render_template('validate.html', stat_auth=stat_auth, key=key)

        return render_template('validate.html', stat_auth="No valid file", key="")


@app.route('/create_backup', methods=['GET'])
def create_backup():
    zipObj = ZipFile('backup.zip', 'w')
    zipObj.write('data.pt')
    zipObj.write('app.py')
    zipObj.write('fake_face.py')
    zipObj.write('ml.py')
    zipObj.close()
    return send_file('backup.zip', as_attachment=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=False)
