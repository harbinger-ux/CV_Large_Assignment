import cv2
import os
from models.ClassicAruco import ClassicAruco
from models.HybridAruco import HybridAruco



if __name__ == "__main__":
    
    TEST_IMAGE = "data/val/000000001171.jpg" 
    DATA_DIR = "data/val"
    YOLO_MODEL_PATH = "weight/det_luma_bnc_s.pt" 


    classic_detector = ClassicAruco(
        max_error=4
    )

    classic_preds = classic_detector.detect(TEST_IMAGE)
    print(f"Classic found {len(classic_preds)} markers")
    #classic_detector.detect_dataset(DATA_DIR, "predictions_classic.csv")

    hybrid_detector = HybridAruco(
        model_path=YOLO_MODEL_PATH,
        max_error=4,
        conf_thresh=0.15,
        iou_thresh=0.5
    )

    hybrid_preds = hybrid_detector.detect(TEST_IMAGE)
    print(f"Hybrid found {len(hybrid_preds)} markers")
    #hybrid_detector.detect_dataset(DATA_DIR, "predictions_hybrid.csv")
