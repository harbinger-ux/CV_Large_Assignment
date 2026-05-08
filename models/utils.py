import cv2
import numpy as np

def get_vectorized_dictionary():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)
    dict_matrices = []
    for i in range(250):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, 160)
        inner_grid = marker_img[20:140, 20:140]
        blocks = inner_grid.reshape(6, 20, 6, 20)
        cores = blocks[:, 10:11, :, 10:11] 
        binary_matrix = (cores > 127).astype(int).reshape(6, 6)
        dict_matrices.append(binary_matrix.flatten())
    return np.array(dict_matrices)

def order_points(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices]

def is_valid_border(warped_bin):
    mask = np.zeros((160, 160), dtype=bool)
    mask[5:15, 5:155] = True
    mask[145:155, 5:155] = True
    mask[15:145, 5:15] = True
    mask[15:145, 145:155] = True
    return np.mean(warped_bin[mask]) <= 75

def extract_and_match(warped_bin_img, dict_tensor):
    inner_grid = warped_bin_img[20:140, 20:140]
    blocks = inner_grid.reshape(6, 20, 6, 20)
    safe_cores = blocks[:, 5:15, :, 5:15]
    medians = np.median(safe_cores, axis=(1, 3))
    extracted_matrix = (medians > 127).astype(int)
            
    min_error = 999
    best_match_id = None
    best_rotation = 0
    
    for rot in range(4):
        rotated_flat = np.rot90(extracted_matrix, k=rot).flatten()
        errors = np.sum(rotated_flat != dict_tensor, axis=1)
        current_min_idx = np.argmin(errors)
        current_min_err = errors[current_min_idx]
        
        if current_min_err < min_error:
            min_error = current_min_err
            best_match_id = current_min_idx
            best_rotation = rot
            
    return min_error, best_match_id, best_rotation

def apply_gamma_correction(image, gamma=3.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_flat_field(gray_img):
    light_map = cv2.GaussianBlur(gray_img, (99, 99), 0)
    normalized = cv2.divide(gray_img, light_map, scale=255)
    return normalized

def apply_log_transform(gray_img):
    img_float = gray_img.astype(np.float32)
    c = 255.0 / np.log(1.0 + np.max(img_float))
    log_image = c * np.log(1.0 + img_float)
    return np.uint8(log_image)
