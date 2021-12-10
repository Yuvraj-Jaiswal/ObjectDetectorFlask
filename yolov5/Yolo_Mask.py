import cv2
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size,non_max_suppression,scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages , LoadStreams
from yolov5.utils.plots import plot_one_box
import torch
import time

img_path = "yolov5/data/images/bus.jpg"
# vid_path = "D:\Mask_Video"
model_path = "yolov5s-VOC.pt"
model_path2 = "Covid_Mask.pt"

imgsz = 640
device = select_device("cpu")
half = device.type != 'cpu'
model = attempt_load(model_path,map_location=device)

stride = int(model.stride.max())
# imgsz = check_img_size(imgsz, s=stride)
names = model.module.names if hasattr(model, 'module') else model.names

colors = [(255,0,0) , (0,255,0) ,  (0,0,255)]

Images = LoadImages(img_path, img_size=imgsz, stride=stride)
# Images = LoadStreams("0", img_size=imgsz, stride=stride)

for path, img, im0s, vid_cap in Images:  # for each image
    Draw_img = im0s
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img = img / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    Detection = model(img,False)[0]
    Detection = non_max_suppression(Detection,0.5, 0.45,None,False, max_det=10)

    for i, det in enumerate(Detection):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], Draw_img.shape).round()
            for *coor , conf , cls in reversed(det):
                plot_one_box(coor,Draw_img,colors[0],label=names[int(cls)],line_thickness=2)

    cv2.imshow("win" , Draw_img)


    if cv2.waitKey(1000)==ord(' '):
        break

cv2.destroyAllWindows()