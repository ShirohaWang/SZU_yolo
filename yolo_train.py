import swanlab
from ultralytics import YOLO

def main():
    # SwanLab 初始化
    swanlab.init(
        project="oral_math_recognition",
        experiment_name="yolov8n_aug_train",
        mode="local"
    )

    # 定义回调函数，在每个 epoch 结束时记录指标
    def on_train_epoch_end(trainer):
        # trainer.metrics 内包含每个 epoch 的指标
        metrics = trainer.metrics
        epoch = trainer.epoch + 1  # 当前 epoch (从 0 开始计数)
        swanlab.log({
            "epoch": epoch,
            "train/box_loss": metrics.get("train/box_loss", 0),
            "train/cls_loss": metrics.get("train/cls_loss", 0),
            "train/dfl_loss": metrics.get("train/dfl_loss", 0),
            "val/precision": metrics.get("metrics/precision(B)", 0),
            "val/recall": metrics.get("metrics/recall(B)", 0),
            "val/mAP50": metrics.get("metrics/mAP50(B)", 0),
            "val/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0)
        }, step=epoch)

    # 加载模型
    model = YOLO("C:/Users/18023/Desktop/yolo_project/weights/yolov8n.pt")

    # 注册回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/exp_aug",
        name="yolov8m_aug_train",
        workers=0
    )

    #训练完成后，记录最终验证结果
    final_metrics = results.results_dict()
    swanlab.log({
        "final/precision": final_metrics.get("metrics/precision(B)", 0),
        "final/recall": final_metrics.get("metrics/recall(B)", 0),
        "final/mAP50": final_metrics.get("metrics/mAP50(B)", 0),
        "final/mAP50-95": final_metrics.get("metrics/mAP50-95(B)", 0)
    })

    # 结束实验
    swanlab.finish()


if __name__ == "__main__":
    main()
