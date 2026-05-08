import argparse
import csv
import math
import os
import cv2
import numpy as np
from collections import defaultdict

class ArucoEvaluator:
    def __init__(self, gt_csv, pred_csv, image_dir, result_csv):
        self.gt_csv = gt_csv
        self.pred_csv = pred_csv
        self.image_dir = image_dir
        self.result_csv = result_csv
        
        self.gt_data = self._parse_csv_to_dict(self.gt_csv)
        self.pred_data = self._parse_csv_to_dict(self.pred_csv)

    def _parse_csv_to_dict(self, filepath):
        """Reads the CSV and maps image_id -> {marker_id: [(x, y), ...]}"""
        data = {}
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found.")
            return data
            
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) 
            
            for row in reader:
                if not row: continue
                image_id = row[0]
                markers = defaultdict(list)
                
                if len(row) > 1 and row[1].strip():
                    parts = row[1].strip().split()
                    for i in range(0, len(parts), 3):
                        m_id = int(parts[i])
                        x, y = float(parts[i+1]), float(parts[i+2])
                        markers[m_id].append((x, y))
                
                data[image_id] = markers
        return data

    def evaluate(self, sigma=0.02, lmbda=1.0):
        per_image_scores = []
        
        with open(self.result_csv, mode='w', newline='', encoding='utf-8') as res_file:
            writer = csv.writer(res_file)
            writer.writerow(['image_id', 'score', 'gt_count', 'spam_count'])
            
            for image_id, gt_markers in self.gt_data.items():
                image_path = os.path.join(self.image_dir, f"{image_id}.jpg") 
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Error: Could not load image {image_path}")
                    continue
                    
                h, w = image.shape[:2]
                diagonal = math.hypot(w, h)
                
                N_gt = sum(len(pts) for pts in gt_markers.values())
                pred_markers = self.pred_data.get(image_id, defaultdict(list))
                N_pred = sum(len(pts) for pts in pred_markers.values())
                
                if N_gt == 0:
                    score = 1.0 if N_pred == 0 else 0.0
                    per_image_scores.append(score)
                    writer.writerow([image_id, score, N_gt, N_pred])
                    continue
                    
                N_spam = 0
                sum_phi = 0.0
                
                for k, p_k in pred_markers.items():
                    g_k = gt_markers.get(k, [])
                    if not g_k:
                        N_spam += len(p_k)
                        continue
                        
                    distances_norm = sorted([min(math.dist(p, g) for g in g_k) / diagonal for p in p_k])
                    valid_matches = distances_norm[:len(g_k)]
                    
                    N_spam += len(distances_norm) - len(valid_matches)
                    sum_phi += sum(math.exp(-(d_norm**2) / (2 * sigma**2)) for d_norm in valid_matches)
                        
                score_img = sum_phi / (N_gt + lmbda * N_spam)
                per_image_scores.append(score_img)
                writer.writerow([image_id, score_img, N_gt, N_spam])
                print(f"[{image_id}] Score: {score_img:.4f} | GT: {N_gt} | Spam: {N_spam}")

        final_score = sum(per_image_scores) / len(per_image_scores) if per_image_scores else 0.0
        print("-" * 40)
        print(f"FINAL SCORE: {final_score:.4f}")
        return final_score

    def visualize_failures(self, upper_thresh=0.2, lower_thresh=0.0):
        if not os.path.exists(self.result_csv):
            print(f"Error: {self.result_csv} not found. Run evaluate() first.")
            return

        with open(self.result_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = float(row['score'])
                if lower_thresh <= score < upper_thresh:
                    image_id = row['image_id']
                    image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
                    image = cv2.imread(image_path)
                    
                    if image is None: continue
                    
                    img_pred, img_gt = image.copy(), image.copy()
                    
                    self._draw_annotations(img_pred, self.pred_data.get(image_id, {}), "PREDICTION", (0, 0, 255))
                    self._draw_annotations(img_gt, self.gt_data.get(image_id, {}), "GROUND TRUTH", (0, 255, 0))
                    
                    combined = np.hstack((img_pred, img_gt))
                    h, w = combined.shape[:2]
                    combined = cv2.resize(combined, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
                    
                    window_name = f"Failed: {image_id} | Score: {score:.4f} | Press ESC to exit"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, w, h)
                    cv2.imshow(window_name, combined)
                    
                    if cv2.waitKey(0) & 0xFF == 27: 
                        break
        cv2.destroyAllWindows()

    def _draw_annotations(self, img, data, title, color):
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        for m_id, points in data.items():
            for x, y in points:
                pt = (int(x), int(y))
                cv2.circle(img, pt, 5, color, -1) 
                cv2.putText(img, f"ID:{m_id}", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate ArUco marker predictions against Ground Truth.")
    parser.add_argument(
        "--data_type", 
        type=str, 
        default="val", 
        help="The dataset split to evaluate (e.g., 'val', 'test')"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="hybrid", 
        help="The model type that generated the predictions (e.g., 'hybrid', 'classic', 'seq')"
    )
    
    args = parser.parse_args()
    evaluator = ArucoEvaluator(
        gt_csv=f"data/{args.data_type}.csv",
        pred_csv=f"predictions_{args.model_type}.csv",
        image_dir=f"data/{args.data_type}/",
        result_csv=f"result_{args.model_type}.csv"
    )
    
    evaluator.evaluate()
    #evaluator.visualize_failures(upper_thresh=0.7, lower_thresh=0.0)