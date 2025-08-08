import numpy as np
import cv2
from PIL import Image
import io

def draw_digit_and_get_pil():
    print("Draw a digit in the window. Press 'q' to finish, 'c' to clear")
    
    canvas_size = 280
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    drawing = False
    
    def draw_circle(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(canvas, (x, y), 10, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow('Draw Digit')
    cv2.setMouseCallback('Draw Digit', draw_circle)
    
    while True:
        cv2.imshow('Draw Digit', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    cv2.destroyAllWindows()
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    pil_img = Image.fromarray(resized, mode='L')
    
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    png_image = Image.open(img_buffer)
    
    return png_image

if __name__ == "__main__":
    pil_image = draw_digit_and_get_pil()
    
    print(f"Type: {type(pil_image)}")
    print(f"Format: {pil_image.format}")
    print(f"Size: {pil_image.size}")