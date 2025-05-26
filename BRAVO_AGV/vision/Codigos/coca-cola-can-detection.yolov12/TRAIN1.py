from ultralytics import YOLO
import gc
import os

import torch

if __name__ == '__main__':
        # Configuración para evitar fragmentación de memoria
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    gc.collect()

    print("✅ Memoria limpia. Modelo listo para ejecutarse.")
    print("GPU detectada:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolo12s.pt")  # Usa los pesos preentrenados

    model.train(
        data="C:/BRAVO/coca-cola-can-detection.v1i.yolov12/data.yaml",
        epochs=500,
        imgsz=480,
        batch=16,
        device=0,
        workers=0,
        name="EVO1",
        project="runs/train",
        val=True,
        plots=False,
        show=False,
        visualize=False,
        resume=False,
        save=True,
    
        
    )
