import cv2  
 
# Initialize webcam (index 0)  
cap = cv2.VideoCapture(0)  
 
# Check if camera opened successfully  
if not cap.isOpened():  
    print("Error: Could not open camera.")  
    exit()  
 
while True:  
    ret, frame = cap.read()  
 
    if not ret:  
        print("Error: Failed to grab frame.")  
        break  
 
    cv2.imshow("Webcam Feed", frame)  
 
    # Exit on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
 
# Cleanup  
cap.release()  
cv2.destroyAllWindows()  

