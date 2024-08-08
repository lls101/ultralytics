from ray import tune
from ultralytics import YOLO
from ray.tune.schedulers import ASHAScheduler
import os

os.environ['WANDB_DIR'] = 'E:/wandb'

# 定义 YOLO 模型
model = YOLO("yolov8n.pt")

# 配置 ASHAScheduler
# 在模型上运行 Ray Tune
result_grid = model.tune(
    data="coco128.yaml",
    space={
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.0, 0.95),
        "box": tune.uniform(0.02, 0.2),
        "cls": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
        "mixup": tune.uniform(0.0, 1.0),
        "copy_paste": tune.uniform(0.0, 1.0)
    },
    epochs=50,
    use_ray=True,
    device='cuda:0' 

)

print(result_grid)
