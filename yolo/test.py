from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Predict_draw_predictions(id_to_class, image_path, output_path, weight_path):
    # Load the trained YOLO segmentation model
    model = YOLO(weight_path)

    # Assign a fixed color to each class
    np.random.seed(42)  
    class_colors = {cls_id: tuple(np.random.randint(0, 255, 3).tolist()) for cls_id in id_to_class}

    # Perform inference
    results = model(image_path)

    # Extract predictions
    predictions = results[0]  
    masks = predictions.masks  
    boxes = predictions.boxes  
    image = cv2.imread(image_path)  

    # Create a blank canvas to overlay masks
    mask_canvas = np.zeros_like(image, dtype=np.uint8)

    # Iterate over masks and draw them on the canvas
    for mask, box in zip(masks.data, boxes):
        cls = int(box.cls[0])  
        class_name = id_to_class.get(cls, "Unknown")  
        color = class_colors[cls]  

        # Convert mask from float to binary and resize to the image dimensions
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

        # Apply color to the mask and overlay on the canvas
        colored_mask = cv2.merge((binary_mask_resized * color[0], 
                                binary_mask_resized * color[1], 
                                binary_mask_resized * color[2]))
        mask_canvas = cv2.addWeighted(mask_canvas, 1, colored_mask, 0.9, 0)

        # Add class name text to the image
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]  
        text_x = max(x1, 0)  
        text_y = max(y1 - 10, 0)  
        cv2.putText(image, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Overlay the segmentation mask on the original image
    mask_canvas = cv2.normalize(mask_canvas, None, 0, 255, cv2.NORM_MINMAX)
    segmented_image = cv2.addWeighted(image, 0.8, mask_canvas, 0.6, 0)

    # Save and display the output image
    cv2.imwrite(output_path, segmented_image)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))  
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Model path
    weight_path = r"results\best.pt"

    # Path to the input image
    image_path = "im1.png"  
    output_path = "output_image.jpg"  

    # Class names dictionary
    yolo_train_ids = {
        'person': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'caravan': 5,
        'trailer': 6,
        'train': 7,
        'motorcycle': 8,
        'bicycle': 9,
    }
    id_to_class = {v: k for k, v in yolo_train_ids.items()}
    Predict_draw_predictions(id_to_class, image_path, output_path, weight_path)