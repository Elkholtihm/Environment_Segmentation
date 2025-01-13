import cv2
import numpy as np



def extract_polygones(mask_path, classes_dict=False):
    # Read the instance mask
    instance_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # Extract unique instance IDs
    unique_instances = np.unique(instance_mask)
    
    # YOLO annotations
    H, W = instance_mask.shape
    annotations = []
    
    for instance_id in unique_instances:
        if instance_id < 1000:  # Ignore non-class pixels
            continue
        
        # Extract class ID and instance ID
        class_id = instance_id // 1000
    
        # Extract binary mask for the current instance
        binary_mask = (instance_mask == instance_id).astype(np.uint8) * 255
    
        # Find contours for the current instance
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:  # Filter small areas
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                if classes_dict:
                    updated_class_id = classes_dict[class_id]
                    annotations.append((updated_class_id, polygon))
                else:
                    annotations.append((class_id, polygon))
    return annotations