from ultralytics import YOLO
import os
import cv2

# モデル読み込み
model = YOLO('yolov8n.pt')
# 推論実行
results = model('https://ultralytics.com/images/bus.jpg')

# 結果取り出し
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    probs = result.probs  # Class probabilities for classification outputs
    print(f'boxes: {boxes}')
    print(f'probs: {probs}')
# 画像削除
try:
    os.remove("./busResults.jpg")
except Exception as e:
    pass


# 結果画像出力
res_plotted = results[0].plot()
cv2.imshow('image', res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
