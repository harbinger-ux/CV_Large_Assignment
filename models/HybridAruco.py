import cv2
import numpy as np
import os
import csv
import math
from ultralytics import YOLO
from models.utils import (get_vectorized_dictionary, order_points, is_valid_border, 
                   extract_and_match, apply_gamma_correction, apply_flat_field)

class HybridAruco:
    def __init__(self, model_path, max_error=4, use_original=True, use_flat_field=True, use_gamma=True, gamma_val=3.0, iou_thresh=0.5, conf_thresh=0.10):
        self.max_error = max_error
        self.use_original = use_original
        self.use_flat_field = use_flat_field
        self.use_gamma = use_gamma
        self.gamma_val = gamma_val
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        
        self.dict_tensor = get_vectorized_dictionary()
        self.detector = YOLO(model_path)

    def detect(self, image_input):
        """Processes a single image (path or cv2 object) using YOLO bounding boxes and returns a list of (id, x, y) tuples."""
        if isinstance(image_input, str):
            pic = cv2.imread(image_input)
        else:
            pic = image_input
            
        if pic is None: 
            return []
            
        gray_orig = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) if len(pic.shape) == 3 else pic
        
        image_variations = []
        if self.use_original: image_variations.append(gray_orig)
        if self.use_flat_field: image_variations.append(apply_flat_field(gray_orig))
        if self.use_gamma: image_variations.append(apply_gamma_correction(gray_orig, gamma=self.gamma_val))

        detections = self.detector(pic, verbose=False, iou=self.iou_thresh, conf=self.conf_thresh)[0].cpu().boxes
        
        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det.xyxy.cpu().numpy()[0]]
            bboxes.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

        found_markers = {}

        for bbox in bboxes:
            x, y, w, h = cv2.boundingRect(bbox.astype(int))
            pad = 12
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(gray_orig.shape[1], x + w + pad), min(gray_orig.shape[0], y + h + pad)
            
            if x2 <= x1 or y2 <= y1: continue
                
            ml_cx, ml_cy = (x - x1) + w / 2.0, (y - y1) + h / 2.0
            
            for current_full_gray in image_variations:
                current_roi = current_full_gray[y1:y2, x1:x2]
                
                for win_size in range(3, 237, 6):
                    thresh = cv2.adaptiveThreshold(current_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win_size, 5)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 75 or area > (current_roi.shape[0] * current_roi.shape[1] * 0.8): continue
                        
                        M_moments = cv2.moments(cnt)
                        if M_moments["m00"] == 0: continue
                        cx, cy = M_moments["m10"] / M_moments["m00"], M_moments["m01"] / M_moments["m00"]
                        
                        if math.dist((cx, cy), (ml_cx, ml_cy)) > max(w, h) * 0.75: continue
                        
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True) 

                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            pts = approx.reshape(4, 2).astype(np.float32)
                            pts += [x1, y1]
                            rect = order_points(pts)
                            
                            dst = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], dtype="float32")
                            M_trans = cv2.getPerspectiveTransform(rect, dst)
                            warped = cv2.warpPerspective(current_full_gray, M_trans, (160, 160))
                            
                            _, warped_otsu = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            
                            min_error, best_match_id, best_rotation = 999, None, 0
                            
                            if is_valid_border(warped_otsu):
                                min_error, best_match_id, best_rotation = extract_and_match(warped_otsu, self.dict_tensor)
                                
                            if min_error > self.max_error:
                                warped_adapt = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
                                if is_valid_border(warped_adapt):
                                    adapt_err, adapt_id, adapt_rot = extract_and_match(warped_adapt, self.dict_tensor)
                                    if adapt_err < min_error:
                                        min_error, best_match_id, best_rotation = adapt_err, adapt_id, adapt_rot
                                        
                            if min_error <= self.max_error:
                                corrected_rect = np.roll(rect, -best_rotation, axis=0)
                                top_left_x, top_left_y = corrected_rect[0]
                                
                                if best_match_id not in found_markers:
                                    found_markers[best_match_id] = []
                                    
                                is_duplicate = False
                                for i, (ex_x, ex_y, ex_err, count) in enumerate(found_markers[best_match_id]):
                                    if math.dist((top_left_x, top_left_y), (ex_x, ex_y)) < 50.0:
                                        is_duplicate = True
                                        if min_error < ex_err:
                                            found_markers[best_match_id][i] = (top_left_x, top_left_y, min_error, 1)
                                        elif min_error == ex_err:
                                            new_x = ((ex_x * count) + top_left_x) / (count + 1)
                                            new_y = ((ex_y * count) + top_left_y) / (count + 1)
                                            found_markers[best_match_id][i] = (new_x, new_y, min_error, count + 1)
                                        break
                                        
                                if not is_duplicate:
                                    found_markers[best_match_id].append((top_left_x, top_left_y, min_error, 1))

        valid_predictions = []
        for marker_id, instances in found_markers.items():
            for (x, y, _, _) in instances:
                valid_predictions.append((marker_id, float(x), float(y)))
                
        return valid_predictions

    def detect_dataset(self, image_dir, output_csv):
        """Calls detect() on a directory of images and formats the results into a CSV."""
        images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'prediction_string'])
            
            for image in images:
                image_id = os.path.splitext(image)[0]
                
                # Call the main detect method
                raw_results = self.detect(os.path.join(image_dir, image))
                
                # Format tuples into string array
                predictions = [f"{res[0]} {res[1]:.3f} {res[2]:.3f}" for res in raw_results]
                prediction_string = " ".join(predictions) if predictions else " "
                
                writer.writerow([image_id, prediction_string])
                print(f"Processed {image_id}: found {len(raw_results)} markers")