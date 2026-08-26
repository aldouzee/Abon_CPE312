import os
import cv2
import numpy as np

def process_digits_image(image_path, output_dir="mnist_custom_dataset"):
    # Create main output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce paper texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding: Convert black ink on white paper to binary (white digit on black background)
    # Using OTSU thresholding combined with BINARY_INV
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of individual digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small noise artifacts by area
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Get bounding boxes for each contour
    boxes = [cv2.boundingRect(cnt) for cnt in valid_contours]
    
    # Sort bounding boxes top-to-bottom, then left-to-right
    # Since the image is a grid (10 rows for digits 0-9), we group into rows by Y-coordinate
    boxes = sorted(boxes, key=lambda b: b[1]) # Sort primarily by Y
    
    # Group bounding boxes into 10 rows
    rows = []
    current_row = []
    row_threshold = 40  # Vertical threshold to determine same row (adjust if needed)
    
    for box in boxes:
        if not current_row:
            current_row.append(box)
        else:
            # If vertical distance to first element in row is small, add to current row
            if abs(box[1] - current_row[0][1]) < row_threshold:
                current_row.append(box)
            else:
                # Sort the completed row left-to-right by X-coordinate
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))
        
    print(f"Detected {len(rows)} rows of digits.")
    
    # Process each row (assuming rows 0 through 9 correspond to digits 0 to 9)
    total_saved = 0
    for digit_label, row in enumerate(rows):
        # Create digit-specific directory (0, 1, 2, ..., 9)
        digit_dir = os.path.join(output_dir, str(digit_label))
        os.makedirs(digit_dir, exist_ok=True)
        
        for idx, (x, y, w, h) in enumerate(row):
            # Extract ROI (Region of Interest) from the inverted binary image
            roi = thresh[y:y+h, x:x+w]
            
            # Make ROI square by padding with black pixels (preserving aspect ratio)
            max_dim = max(w, h)
            square_roi = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Center the digit in the square box
            offset_x = (max_dim - w) // 2
            offset_y = (max_dim - h) // 2
            square_roi[offset_y:offset_y+h, offset_x:offset_x+w] = roi
            
            # Add additional padding around the digit before resizing to mimic MNIST style
            pad = int(max_dim * 0.2)
            padded_roi = cv2.copyMakeBorder(
                square_roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
            )
            
            # Resize image to standard MNIST 28x28 dimensions
            resized_digit = cv2.resize(padded_roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Save final image
            file_name = f"digit_{digit_label}_sample_{idx+1}.png"
            file_path = os.path.join(digit_dir, file_name)
            cv2.imwrite(file_path, resized_digit)
            total_saved += 1

    print(f"Successfully processed and saved {total_saved} digit images to '{output_dir}'.")

# Run script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_filename = os.path.join(script_dir, "digits_image.jpg")
    output_dir = os.path.join(script_dir, "mnist_custom_dataset")

    if not os.path.exists(image_filename):
        print(f"Error: '{image_filename}' not found. Please provide the input image next to this script.")
    else:
        process_digits_image(image_filename, output_dir)