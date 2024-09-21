from ultralytics import YOLOv10,YOLO
import torch

if __name__=="__main__":
    #DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(DEVICE)
    URL = r'C:\Users\kawaw\yolo\yolo_dataset\yolov10n.pt'
    path_yaml = r"C:\Users\kawaw\yolo\yolo_dataset\dataset.yaml"
    #yolov8
    torch.cuda.set_device(0)
    model = YOLO(URL)
    # Train the model on your custom dataset
    results = model.train(
        data=path_yaml,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=8,
        device=DEVICE,
        patience=0
    )
    savepath=r'C:\Users\kawaw\yolo\yolo_dataset\fine_tuned_model.pt'
    model.save(savepath)