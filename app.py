from flask import Flask,render_template,request
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression,scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages
from yolov5.utils.plots import plot_one_box
from PIL import Image
import random
import torch
import os

random.seed(10)

vid_path = "D:\Mask_Video"
weight_OD = "yolov5s.pt"
weight_Mk = "Covid_Mask.pt"
imgsz = 640
device = select_device("cpu")
model_MK = attempt_load(weight_Mk,map_location=device)
model_OD = attempt_load(weight_OD,map_location=device)
stride_OD = int(model_OD.stride.max())
stride_MK = int(model_MK.stride.max())
names_OD = model_OD.module.names if hasattr(model_OD, 'module') else model_OD.names
names_MK = model_MK.module.names if hasattr(model_MK, 'module') else model_MK.names
colors = []
for _ in range(200):
    colors.append( (random.randint(0,255),random.randint(0,205),random.randint(0,255) ) )

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def GetImgYoloOD(pth):
    numDetected = {}
    Image = LoadImages(pth, img_size=imgsz, stride=stride_OD)
    Draw_img = "ref"
    for path, img, im0s, vid_cap in Image:  # for each image
        Draw_img = im0s
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img = img / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        Detection = model_OD(img, False)[0]
        Detection = non_max_suppression(Detection, 0.25, 0.45, None, False, max_det=100)

        DetectionM = model_MK(img, False)[0]
        DetectionM = non_max_suppression(DetectionM, 0.25, 0.45, None, False, max_det=100)

        for i, det in enumerate(DetectionM):  # detections per image

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], Draw_img.shape).round()
                for *coor, conf, cls in reversed(det):
                    clor = int(cls)
                    if cls==1:clor = 2
                    if cls==2:clor = 1
                    plot_one_box(coor, Draw_img, colors[int(clor)], label=names_MK[int(cls)], line_thickness=2)
                    if names_MK[int(cls)] not in numDetected.keys():
                        numDetected.update({ str(names_MK[int(cls)]) : 1 })
                    else:
                        numDetected[str(names_MK[int(cls)])] += 1

        for i, det in enumerate(Detection):  # detections per image

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], Draw_img.shape).round()
                for *coor, conf, cls in reversed(det):
                    plot_one_box(coor, Draw_img, colors[int(cls)], label=names_OD[int(cls)], line_thickness=2)
                    if names_OD[int(cls)] not in numDetected.keys():
                        numDetected.update({ str(names_OD[int(cls)]) : 1 })
                    else:
                        numDetected[str(names_OD[int(cls)])] += 1

    return Draw_img , numDetected



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.getcwd()
        file_path = os.path.join(
            basepath, 'uploads',f.filename)
        f.save(file_path)

        pth = "uploads/" + f.filename
        img , dic = GetImgYoloOD(pth)
        img = img[... , ::-1]
        img = Image.fromarray(img)
        img.save("static/predicted/" + f.filename)
        value = str(dic)
        value = value[1:len(value)-1]
        path, dirs, files = next(os.walk("uploads"))
        if len(files) > 20 :
            os.remove(f"uploads/{files[0]}")

        path_p, dirs_p, files_p = next(os.walk("static/predicted"))
        if len(files_p) > 20:
            os.remove(f"static/predicted/{files_p[0]}")

        print("static/predicted/" + f.filename + "-" + value)
        return "static/predicted/" + f.filename + "-" + value

    return None


if __name__ == '__main__':
    # app.debug = True
    app.run()
