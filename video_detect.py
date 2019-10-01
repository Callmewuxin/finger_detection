# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time
from keras.models import load_model
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# initialize the video stream and allow the camera sensor to warm up
print('[INFO] camera sensor warming up...')
vs = VideoStream(usePiCamera=-1 > 0).start()
time.sleep(2.0)

def loadmodel():
    print('start load model')
    global model
    model_dir = 'finger_detection_model.h5'
    if os.path.exists(model_dir):
        print('yes it is')
        model = load_model(model_dir)
    else:
        print('it doesnt')
    global graph
    graph = tf.get_default_graph()

loadmodel()
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    img_read = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_read = cv2.resize(img_read, (128, 128))
    predict_mat = []
    predict_mat.append(img_read)
    predict_mat = np.array(predict_mat)
    predict_mat = predict_mat.reshape(1, 128, 128, 1)
    with graph.as_default():
        preds = model.predict(predict_mat, steps=1)
        predicted_class_indices = np.argmax(preds, axis=1)
        prediction = predicted_class_indices[0]

    cv2.putText(frame, "number: {:d}".format(prediction), (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()