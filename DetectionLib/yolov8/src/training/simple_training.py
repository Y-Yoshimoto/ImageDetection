#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
# 時間計測用
import time
# OpenCV
import cv2
# ディレクトリ,ファイル操作用
import os
import glob
# yaml読み込み
import yaml
# 標準出力抑止
import contextlib
import io

class TrainYolov8:
    def __init__(self, data: str="./data.yaml", baseModel:str="yolov8n.pt"):
        """ 学習用のYOLOv8クラス """
        self.model = YOLO(baseModel)
        self.data = data
        
    ###################### 学習ワークフロー ######################
    def trainSinpleFlows(self, epochs: int=100, patience: int=20, batch:int=3, project: str="project", exist_ok: bool=False):
        """ 学習評価出力フロー """
        self.project = project
        # 学習
        ptime = self._processing_time()
        self.trainsingle(epochs=epochs, batch=batch, val=True, project=project)
        
        # 評価
        metrics = self.val()
        # 評価結果表示
        self._printVal(metrics)
        
        # テストデータの結果画像出力
        self.testPredict()
        
        # ONNX形式でモデルをエクスポート
        #self.exportONNX()
        
        # OpenVINO形式でモデルをエクスポート
        #self.exportOpenVino()
        
        # 時間計測
        totaltime = ptime()
        print(f'Total time={(totaltime) : .3f})[s]')
        print(f'Total time {(totaltime/60) : .3f}[m]')
        print(f'Total time={ (totaltime/3600):.3f} [h]')
        
    def trainsingle(self, epochs: int, batch:int, val: bool, project: str, exist_ok: bool=False):
        """ 学習を実行する """
        self.model.train(data=self.data, epochs=epochs, batch=batch, val=val, project=project, exist_ok=exist_ok)
        
    ###################### 結果評価 ######################
    def val(self, split='val') -> object:
        """ 学習結果を評価する """
        return self.model.val(split=split)
    
    def _printVal(self, metrics):
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

    ###################### モデルのエクスポート ######################
    def exportONNX(self):
        """ ONNX形式でモデルをエクスポートする """
        self.model.export(format="onnx")
        
    def exportOpenVino(self):
        """ ONNX形式でモデルをエクスポートする """
        self.model.export(format="openvino")
        
    ###################### 処理時間計測 ######################
    def _processing_time(self):
        """ 時間を計測するクロージャ """
        start = time.time()
        def function():
            return time. time() - start
        return function
    
    ###################### 結果画像生成 ######################
    def testPredict(self, datakey='test'):
        """ 推論を実行する """
        # 出力用ディレクトリーを作成
        projectDir=f'./{self._train_args("project")}/test'
        os.makedirs(projectDir, exist_ok=True)
        # testデータ抽出
        testImages=self._loadImages(yamlpath=self._train_args("data"), key=datakey)
        # 推論実行/結果保存
        for filepath,filename in testImages:
            #cv2.imwrite(f'{projectDir}/{filename}', self.model.predict(filepath)[0].plot())
            cv2.imwrite(f'{projectDir}/{filename}', self._silentPredict(filepath)[0].plot())

    def _loadAnnotationDir(self, yamlpath: str="./data.yaml",key='test'):
        """ アノテーションデータの保存場所を返す """
        with open(yamlpath) as file:
            data_dict = yaml.load(file, Loader=yaml.FullLoader)
        return data_dict[key] # ['train'] or ['val'] or ['test']

    def _loadImages(self, yamlpath: str="./data.yaml",key='test'):
        """ アノテーション画像の一覧を返す """
        ImagePath=self._loadAnnotationDir(yamlpath=yamlpath, key=key)
        return [(i, os.path.basename(i)) for i in glob.glob(ImagePath + '/*.jp*g')]
        
    ###################### 補助関数 ######################
    def _train_args(self, key:str):
        """ 学習時の引数を返す """
        return self.model.ckpt["train_args"][key]
    
    def _silentPredict(self, filepath: str):
        return self.model.predict(filepath,verbose=False)
        #with contextlib.redirect_stdout(io.StringIO()):
        #    return self.model.predict(filepath)

def main():
    """ メイン関数 """
    # 学習データ
    data = "./data.yaml"
    # ベースモデル
    baseModel = "./Detect/train/weights/best.pt"
    # 学習用のYOLOv8クラス
    #trainYolov8 = TrainYolov8(data)
    trainYolov8 = TrainYolov8(data=data,baseModel=baseModel)
    
    # 学習評価出力フロー
    #trainYolov8.trainSinpleFlows(epochs=1, patience=20, project="project2")
    #trainYolov8.trainSinpleFlows(epochs=1, patience=20, project="project2", exist_ok=True)
    trainYolov8.testPredict()
    
if __name__ == '__main__':
    main()