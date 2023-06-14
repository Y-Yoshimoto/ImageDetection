#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC

# 時間計測用
import time
# OpenCV
import cv2
# pytorch
import torch

class PredictYolov8(ABC):
    """ 推論基底クラス """
    def __init__(self, modelfile:str="yolov8n.pt"):
        """ コンストラクタ モデルファイル読み込み """
        # モデル読み込み
        self.model = torch.load(modelfile)
        # 推論モードに設定
        self.model.eval()

    def __call__(self, image):
        """ 推論を実行 """
        return self._silentPredict(image)
    
    ###################### 画像フォーマット変更 ######################
    
    def _toTensor(self, filename: str):
        """ 画像をテンソルに変換する """
        # 画像読み込み
        img = cv2.imread(filename)
        # 画像をテンソルに変換
        return torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        
    
    ###################### 補助関数 ######################
    def _silentPredict(self, filepath: str):
        """ 出力を停止して推論を実行 """
        return self.model.predict(filepath,verbose=False)

    def _train_args(self, key:str):
        """ 学習時の引数を返す """
        return self.model.ckpt["train_args"][key]

    def _processing_time(self):
        """ 時間を計測するクロージャ """
        start = time.time()
        def function():
            return time. time() - start
        return function