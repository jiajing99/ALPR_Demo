import os
import cv2
from flask import Flask, render_template, request
from datetime import timedelta

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# basedir = os.path.abspath(os.path.dirname(__file__))
imagePath1 = "static/images/pic0.jpg"
imagePath2 = "static/images/pic1.jpg"
label = ''
isUpload = False
hasResult = False

@app.route("/")
def index():
    return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)

@app.route("/", methods=["POST"])
def upload():
    global isUpload, label, hasResult
    label = ''
    hasResult = False

    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            isUpload = False
            print('No file part')
            return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
        f = request.files['file']

        # if user does not select file, browser also submit an empty part without filename
        if f.filename == '':
            isUpload = False
            print('No selected file')
            return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)

        isUpload = True
        filePath = "static/images/picture.jpg"
        f.save(filePath)

        img = cv2.imread(filePath)
        img = cv2.resize(img, (650, 450))
        newFilePath = "static/images/resize_picture.jpg"
        cv2.imwrite(newFilePath, img)

        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)

@app.route("/detect", methods=["GET","POST"])
def detect():
    global hasResult, label
    label = ''

    if isUpload==False:
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cd ../pytorch-YOLOv4-master'
        cmd2 = 'CUDA_VISIBLE_DEVICES=5 python models.py 1 Yolov4_epoch500.pth ../Demo/static/images/picture.jpg 608 608 data/plate.names'
        cmd3 = 'cp predictions.jpg ../Demo/static/images'
        cmd4 = 'cp result.jpg ../Demo/static/images'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4)

        filePath = "static/images/predictions.jpg"
        img = cv2.imread(filePath)
        img = cv2.resize(img, (650, 450))
        newFilePath = "static/images/resize_picture.jpg"
        cv2.imwrite(newFilePath, img)

        filePath2 = "static/images/result.jpg"
        img = cv2.imread(filePath2)

        if img.shape[0] == 2 and img.shape[1] == 2:
            hasResult = False
            return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)

        else:
            hasResult = True
            img = cv2.resize(img, (340, 100))
            newFilePath2 = "static/images/resize_result.jpg"
            cv2.imwrite(newFilePath2, img)
            return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

@app.route("/recognize", methods=["GET","POST"])
def recognize():
    global label
    newFilePath = "static/images/resize_picture.jpg"
    newFilePath2 = "static/images/resize_result.jpg"

    if isUpload==False:
        label = ''
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    elif hasResult == False:
        label = ''
        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cp static/images/result.jpg ../CRNN_Chinese_Characters_Rec-stable'
        cmd2 = 'cd ../CRNN_Chinese_Characters_Rec-stable'
        cmd3 = 'CUDA_VISIBLE_DEVICES=5 python demo.py --image_path result.jpg --checkpoint output/OWN/crnn/2021-02-23-16-02/checkpoints/checkpoint_19_acc_0.9982.pth'
        cmd4 = 'cp result.txt ../Demo/static'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4)

        img = cv2.imread(newFilePath2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(newFilePath2, img)

        data = []
        with open('static/result.txt', 'r') as f:
            for line in f.readlines():
                data.append(line[:-1])
            for str in data:
                datalist = str.split()
            label = "NUMBER: " + datalist[-1]

        return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

@app.route("/fgsm1", methods=["GET","POST"])
def fgsm1():
    global label
    newFilePath = "static/images/resize_picture.jpg"
    newFilePath2 = "static/images/adv.jpg"

    if isUpload == False:
        label = ''
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    elif hasResult == False:
        label = ''
        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cp static/images/result.jpg ../CRNN_Chinese_Characters_Rec-stable'
        cmd2 = 'cd ../CRNN_Chinese_Characters_Rec-stable'
        cmd3 = 'CUDA_VISIBLE_DEVICES=5 python fgsmAttack1.py'
        cmd4 = 'cp adv.txt ../Demo/static'
        cmd5 = 'cp adv.jpg ../Demo/static/images'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4 + ';' + cmd5)

        img = cv2.imread(newFilePath2)
        img = cv2.resize(img, (340, 100))
        cv2.imwrite(newFilePath2, img)

        with open('static/adv.txt', 'r') as f:
            label = "NUMBER: " + f.read()

        return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

@app.route("/pgd1", methods=["GET","POST"])
def pgd1():
    global label
    newFilePath = "static/images/resize_picture.jpg"
    newFilePath2 = "static/images/adv.jpg"

    if isUpload == False:
        label = ''
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    elif hasResult == False:
        label = ''
        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cp static/images/result.jpg ../CRNN_Chinese_Characters_Rec-stable'
        cmd2 = 'cd ../CRNN_Chinese_Characters_Rec-stable'
        cmd3 = 'CUDA_VISIBLE_DEVICES=5 python pgdAttack1.py'
        cmd4 = 'cp adv.txt ../Demo/static'
        cmd5 = 'cp adv.jpg ../Demo/static/images'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4 + ';' + cmd5)

        img = cv2.imread(newFilePath2)
        img = cv2.resize(img, (340, 100))
        cv2.imwrite(newFilePath2, img)

        with open('static/adv.txt', 'r') as f:
            label = "NUMBER: " + f.read()

        return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

@app.route("/fgsm2", methods=["GET","POST"])
def fgsm2():
    global label
    newFilePath = "static/images/resize_picture.jpg"
    newFilePath2 = "static/images/adv.jpg"

    if isUpload == False:
        label = ''
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    elif hasResult == False:
        label = ''
        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cp static/images/result.jpg ../CRNN_Chinese_Characters_Rec-stable'
        cmd2 = 'cd ../CRNN_Chinese_Characters_Rec-stable'
        cmd3 = 'CUDA_VISIBLE_DEVICES=5 python fgsmAttack1.py --checkpoint output/OWN/crnn/2021-02-23-19-43/checkpoints/checkpoint_19_acc_0.9956.pth'
        cmd4 = 'cp adv.txt ../Demo/static'
        cmd5 = 'cp adv.jpg ../Demo/static/images'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4 + ';' + cmd5)

        img = cv2.imread(newFilePath2)
        img = cv2.resize(img, (340, 100))
        cv2.imwrite(newFilePath2, img)

        with open('static/adv.txt', 'r') as f:
            label = "NUMBER: " + f.read()

        return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

@app.route("/pgd2", methods=["GET","POST"])
def pgd2():
    global label
    newFilePath = "static/images/resize_picture.jpg"
    newFilePath2 = "static/images/adv.jpg"

    if isUpload == False:
        label = ''
        return render_template('index.html', imagePath1=imagePath1, imagePath2=imagePath2, label=label)
    elif hasResult == False:
        label = ''
        return render_template('index.html', imagePath1=newFilePath, imagePath2=imagePath2, label=label)
    else:
        cmd1 = 'cp static/images/result.jpg ../CRNN_Chinese_Characters_Rec-stable'
        cmd2 = 'cd ../CRNN_Chinese_Characters_Rec-stable'
        cmd3 = 'CUDA_VISIBLE_DEVICES=5 python pgdAttack1.py --checkpoint output/OWN/crnn/2021-02-23-19-43/checkpoints/checkpoint_19_acc_0.9956.pth'
        cmd4 = 'cp adv.txt ../Demo/static'
        cmd5 = 'cp adv.jpg ../Demo/static/images'
        os.system(cmd1 + ';' + cmd2 + ';' + cmd3 + ';' + cmd4 + ';' + cmd5)

        img = cv2.imread(newFilePath2)
        img = cv2.resize(img, (340, 100))
        cv2.imwrite(newFilePath2, img)

        with open('static/adv.txt', 'r') as f:
            label = "NUMBER: " + f.read()

        return render_template('index.html', imagePath1=newFilePath, imagePath2=newFilePath2, label=label)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)  # http://10.112.33.129:8000
