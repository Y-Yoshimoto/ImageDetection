#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
from ultralytics import YOLO
# 時間計測用
import time
# OpenCV
import cv2
# ディレクトリ,ファイル操作用
import os
import glob

class PredictYolov8:
    def __init__(self, baseModel:str="yolov8n.pt"):
        """ YOLOv8クラス """
        self.model = YOLO(baseModel)
        
    def __call__(self, image) -> Any:
        """ 推論を実行する """
        return self._silentPredict(image)
    
    ###################### 処理時間計測 ######################
    def _processing_time(self):
        """ 時間を計測するクロージャ """
        start = time.time()
        def function():
            return time. time() - start
        return function
    
    ###################### 結果画像生成 ######################
    def batchPredict(self, imagePath: str, exportPath: str):
        """ 推論を実行する """
        # 出力用ディレクトリーを作成
        os.makedirs(exportPath, exist_ok=True)
        # testデータ抽出
        images=self._loadImages(path=imagePath)
        # 推論実行/結果保存
        for filepath,filename in images:
            #cv2.imwrite(f'{projectDir}/{filename}', self.model.predict(filepath)[0].plot())
            cv2.imwrite(f'{exportPath}/re_{filename}', self._silentPredict(filepath)[0].plot())

    def _loadImages(self, path: str):
        """ 画像の一覧(フルパス,ファイル名)を返す """        
        files = glob.glob('*.jp*g') + glob.glob('*.png') + glob.glob('*.JP*G') + glob.glob('*.PNG')
        return [(filepath,os.path.basename(filepath)) for filepath in files]
        
    ###################### 補助関数 ######################
    def _train_args(self, key:str):
        """ 学習時の引数を返す """
        return self.model.ckpt["train_args"][key]
    
    def _silentPredict(self, filepath: str):
        return self.model.predict(filepath,verbose=False)

def main():
    """ メイン関数 """
    
if __name__ == '__main__':
    main()