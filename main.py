import cv2
import numpy as np
import onnxruntime as ort
import time


# standard COCO dataset labels taken from Google which contains 80 classes
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def main():
    print("Loading model...")

    options = ort.SessionOptions()
    session = ort.InferenceSession("yolo26n_int8.onnx", sess_options=options, providers=["CPUExecutionProvider"])
    # providers explicitly tells it to run the math on your CPU instead of looking for a graphics card

    # input layer and output layer names
    input_name = session.get_inputs()[0].name  # images
    output_name = session.get_outputs()[0].name  # output0

    #opening web-cam
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not capture.isOpened():
        print("Could not open camera")
        return

    print("System running. Press 'q' inside the view window to exit.")

    while True:
        ret, frame = capture.read()

        #if we are not getting frames then exit loop
        if not ret:
            break

        #re-sizing to 640x640 to match
        display_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        #changing bgr to rgb
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)



        ### now we have to ready the frames(input tensor) to feed NN

        #we need to convert it into CHW (Channel, Height, Width), as model expects that format
        input_tensor = np.transpose(rgb_frame, (2, 0, 1)).astype(np.float32) / 255.0

        #Shape goes from (3, H, W) → (1, 3, H, W).
        #That extra dimension represents the batch size. Here, it means we are sending 1 image
        # Even if you’re processing a single image, models expect a batch dimension.
        input_tensor = np.expand_dims(input_tensor, axis=0)

        ### ------

        # execute inference
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000  # convert to milliseconds
        
        raw_output = np.array(outputs[0])  # remove extra brackets and convert to ndarray
        raw_output = raw_output[0] # removing batch dimension

        # Display latency on frame
        latency_text = f"Inference: {inference_time:.2f}ms"
        cv2.putText(display_frame, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # let's call each single prediction a 'detection'
        # detection = [x1, y1, x2, y2, confidence score, class id]
        for detection in raw_output:
            #safety net: when model outputs incomplete rows due to glitch, we will continue to next detection
            if detection.size < 6:
                continue

            confidence_score = float(detection[4])
            if confidence_score > 0.4:
                x1, y1, x2, y2 = map(int, np.round(detection[0:4]))
                class_id = int(detection[5])
                w, h = x2 - x1, y2 - y1

                if w>0 and h>0:
                    #draw bounding box overlay
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (180, 0, 255), 2)
                    #displaying class of detected object as text
                    class_name = classNames[class_id] if 0 <= class_id < len(classNames) else "unknown"
                    label = f"{class_name} ({confidence_score:.2f})"
                    cv2.putText(display_frame, label,(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 255), 1)


        # show the annotated frame and check for exit
        cv2.imshow("detections", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()