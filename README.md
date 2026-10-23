牙齿及缺陷分割数据集， 2495 张图像，其中有 28904 个标记对象,1GB数据量，json格式标注![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/09697f5d48f24312b424dfa9c1c3f1a9.png)

详细标注一下四类目标
tooth：牙齿；齿状物。
caries：龋齿；骨疡。
cavity：洞；腔；窟窿。
crack：裂缝；裂纹![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d768c79f49f445a1a883dd3c6bfc54ce.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f528c130543e4254bb2392ad2efa3b3f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0ca101c1d11044ba8fe9054415cf138b.png)
# 牙病牙齿实例分割系统 YOLOv11 完整代码
## 环境安装
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

## 1、数据集配置文件 tooth.yaml
```yaml
path: ./tooth_dataset
train: images/train
val: images/val
test: images/test

names:
  0: tooth        # 牙齿
  1: caries       # 龋齿（蛀牙）
```

## 2、训练代码 train_tooth.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11实例分割预训练权重
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")

    results = model.train(
        data="tooth.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="tooth_seg_result",
        name="yolo11n_tooth",
        amp=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
```

## 3、图像推理 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("./tooth_seg_result/yolo11n_tooth/weights/best.pt")

def detect_tooth(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        out_img = r.plot()
        cv2.imwrite("tooth_pred_result.jpg", out_img)
    return out_img

if __name__ == "__main__":
    detect_tooth("dental_test.jpg")
```

## 4、PyQt5可视化界面代码 gui_tooth.py
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class ToothSegUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("口腔牙齿&龋齿实例分割检测系统")
        self.setGeometry(100, 100, 1400, 800)
        self.model = YOLO("./tooth_seg_result/yolo11n_tooth/weights/best.pt")

        self.btn_select = QPushButton("选择口腔图片", self)
        self.btn_select.move(20, 20)
        self.btn_select.clicked.connect(self.load_image)

        self.label_origin = QLabel("原图", self)
        self.label_origin.setGeometry(20, 80, 650, 600)
        self.label_result = QLabel("分割结果", self)
        self.label_result.setGeometry(700, 80, 650, 600)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", ".", "图片(*.jpg *.png *.jpeg)")
        if not file_path:
            return
        img = cv2.imread(file_path)
        pred = self.model(file_path, conf=0.3)[0]
        pred_img = pred.plot()
        self.show_img(img, self.label_origin)
        self.show_img(pred_img, self.label_result)

    def show_img(self, cv_img, label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, w*c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ToothSegUI()
    window.show()
    sys.exit(app.exec_())
```
## 数据集标签说明
|类别编号|类别英文|类别中文|标注说明|
| ---- | ---- | ---- | ---- |
|0|tooth|牙齿|青色外轮廓分割框，单颗牙齿实例分割|
|1|caries|龋齿|红色标记龋坏区域分割掩码|

## 应用场景
1. 口腔内窥镜图片、口内照片AI辅助筛查，分割单颗牙齿，定位蛀牙龋坏区域
2. 牙科辅助诊断，批量口扫图像自动标注，减少医生人工阅片工作量
3. 儿童龋齿筛查、口腔健康普查项目，自动输出龋齿位置与面积评估

运行命令：`python gui_tooth.py`，支持图片检测，可扩展视频、摄像头实时内窥镜检测。


