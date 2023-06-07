#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import time

class TrainYolov8:
    def __init__(self, data: str, baseModel="yolov8n.pt"):
        """ 学習用のYOLOv8クラス """
        self.model = YOLO(baseModel)
        self.data = data
        
    def trainSinpleFlows(self, epochs: int=100, patience: int=20, batch:int=6, project: str="project"):
        """ 学習評価出力フロー """
        
        # 学習
        ptime = self.processing_time()
        self.trainsingle(epochs=epochs, batch=batch, val=True, project=project)
        
        # 評価
        metrics = self.val()
        
        # 結果表示
        self.printVal(metrics)
        
        # ONNX形式でモデルをエクスポート
        #self.exportONNX()
        
        # OpenVINO形式でモデルをエクスポート
        #self.exportOpenVino()
        
        # 時間計測
        totaltime = ptime()
        print(f'Total time={(totaltime) : .3f})[s]')
        print(f'Total time {(totaltime/60) : .3f}[m]')
        print(f'Total time={ (totaltime/3600):.3f} [h]')
        
        
    def trainsingle(self, epochs: int, batch:int, val: bool, project: str):
        """ 学習を実行する """
        self.model.train(data=self.data, epochs=epochs, batch=batch, val=val, project=project)
        

    def val(self) -> object:
        """ 学習結果を評価する """
        return self.model.val()
    
    def printVal(self, metrics):
        """ 学習結果を表示する """
        print("-----------------")
        for i, map in enumerate (metrics.box. maps. tolist ()) :
            print (f'Class[{i}] mAP: {map: .3f} / {self.model.names[i]}')
        print(f'Box mAP50-95: {metrics.box.map: .3f}')
        print(f'Box mAP50: {metrics.box.map50: .3f}')
        print(f'Box mAP75: {metrics.box.map75: .3f}')
        #print(f'Train time={ptraintime: .3f}) [s]')
        #print (f'Validation time { (pvaltime-ptraintime) : .3f3)[s]")
        #print (f'Export time={(totaltime-pvaltime) : .3f3)[s]")
        # print(f'Total time={(totaltime) : .3f})[s]")
        # print (f'Total time (totaltime/60) : .3f}[m]')
        # print (f'Total time={ (totaltime/3600):.3f} [h]')
                
    def exportONNX(self):
        """ ONNX形式でモデルをエクスポートする """
        self.model.export(format="onnx")
        
    def exportOpenVino(self):
        """ ONNX形式でモデルをエクスポートする """
        self.model.export(format="openvino")

    def processing_time(self):
        """ 時間を計測するクロージャ """
        start = time.time()
        def function():
            return time. time() - start
        return function
    
def main():
    """ メイン関数 """
    # 学習データ
    data = "./data.yaml"
    # 学習用のYOLOv8クラス
    trainYolov8 = TrainYolov8(data)
    # 学習評価出力フロー
    trainYolov8.trainSinpleFlows(epochs=100, patience=20)
    
if __name__ == '__main__':
    main()