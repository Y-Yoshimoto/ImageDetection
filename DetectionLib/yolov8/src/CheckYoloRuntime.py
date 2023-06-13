#!/usr/bin/env python
# coding:utf-8
from ultralytics import YOLO
import os
import cv2

# モデル読み込み
model_pt = 'yolov8n.pt'
#model_pt = 'yolov8n-seg.pt'
model = YOLO(model_pt)
# 推論実行
results = model('https://ultralytics.com/images/bus.jpg')

# 結果取り出し
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    #print(f'names: {result.names}')
    #print(f'result:\n {result}')
    #print(f'boxes: {boxes}')
    classes = [int(x) for x in boxes.cls.tolist()]
    conf = boxes.conf.tolist()
    classes_name = [result.names[x] for x in classes]
    print(f'classes: {classes}')
    print(f'conf: {conf}')
    print(f'classes: {classes_name}')
# 画像/ネットワーク削除
try:
    os.remove("./bus.jpg")
    os.remove("./"+model_pt)
except Exception as e:
    pass


# 結果画像出力
res_plotted = results[0].plot()
cv2.imwrite('result.jpg', res_plotted)