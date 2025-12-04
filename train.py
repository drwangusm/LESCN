import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# ['yolo11-LSCD','yolo11-starnet-C3k2-Star-LSCD','yolo11-C3k2-Star-LSCD','yolo11-starnet-C3k2-Star-CAA-LSCD'] 该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
# ['yolo11-starnet-C3k2-Star-CAA-LSCD','yolo11-C3k2-Star-CAA']


if __name__ == '__main__':
    for yaml_name in ['yolo11-LSCD','yolo11-starnet','yolo11-C3k2-Star','yolo11-starnet-C3k2-Star-LSCD','yolo11','yolo11-starnet-C3k2-Star-CAA-LSCD','yolo11-C3k2-Star-CAA']:
        model = YOLO(f'./models/innovations/{yaml_name}.yaml')
        model.train(data='xxx/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,# 默认是16
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp 
                # fraction=0.2,
                project='runs/',
                name=yaml_name,
                )