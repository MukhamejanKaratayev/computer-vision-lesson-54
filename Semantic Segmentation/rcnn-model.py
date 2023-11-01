import cv2
import numpy as np

net = cv2.dnn.readNetFromTensorflow("Semantic Segmentation/frozen_inference_graph.pb","Semantic Segmentation/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt") 

classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
           "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
           "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
           "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
           "baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass",
           "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
           "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","mirror","dining table",
           "window","desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
           "oven","toaster","sink","refrigerator","blender","book","clock","vase","scissors","teddy bear",
           "hair drier","toothbrush"]

print("NUM CLASSES:", len(classes))

font = cv2.FONT_HERSHEY_PLAIN     # font
text_color = (0,255,0)            # green color

# random colors to distinguish between different classes (90 classes, 3 channels)
colors = np.random.randint(0, 255, (90, 3))   # generate 90 random colors

cap = cv2.VideoCapture('Resources/test.mp4')
while True:

    ret, img = cap.read()
    
    if ret == True:
        
        output = img.copy()             # copy of the original frame 
        overlay = img.copy()            # copy of the original frame

        height, width, _ = img.shape    # retrieve shape from image (frame)
        
        blob = cv2.dnn.blobFromImage(img, swapRB=True)  
        
        net.setInput(blob)                                                     
        boxes, masks = net.forward(["detection_out_final", "detection_masks"]) 
                                                                               

        num_detections = boxes.shape[2]        

        for i in range(num_detections):        

            box = boxes[0,0,i]                
            class_id = int(box[1])            
            confidence_score = box[2]         

            if confidence_score > 0.5:       
                           
                label = str(classes[class_id])    

                x1, y1, x2, y2 = box[3:]          

                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)

                object_area = overlay[y1:y2, x1:x2]    

                object_height, object_width, _ = object_area.shape   

                mask = masks[i, class_id]    
                mask = cv2.resize(mask, (object_width, object_height))

                _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

                mask = np.array(mask, np.uint8)     # convert to array of integer 
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:     # for each contour...
                    cv2.fillPoly(object_area,            # area to fill
                                 [cnt],                  # contours bounding the area
                                 (int(colors[class_id][0]), int(colors[class_id][1]), int(colors[class_id][2]))# color
                                 )
                
                # We put text (class and confidence score) on top of each detected object
                cv2.putText(output, label + " " + str(round(confidence_score,2)), (x1, y1), font, 1.2, 
                            text_color, 2)   # text of the box 


        alpha = 0.6
        cv2.addWeighted(overlay,  # image that we want to “overlay” on top of the original image
                        alpha,    # alpha transparency of the overlay (the closer to 0 the more transparent the 
                                  # overlay will appear)
                        output,   # original source image
                        1-alpha,  # beta parameter (1-alpha)
                        0,        # gamma value — a scalar added to the weighted sum (we set it to 0)
                        output    # our final output image
                        )
            
        cv2.imshow("out", output)    
    
    
    else: 
        break
    

    cv2.imshow('Input',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break