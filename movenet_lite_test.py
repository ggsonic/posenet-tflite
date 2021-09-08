import cv2
import tensorflow as tf

# Load the input image.
image_path = 'sample.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.image.resize_with_pad(image, 256, 256)

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# TF Lite format expects tensor type of float32.
#input_image = tf.cast(image, dtype=tf.float32)
input_image = tf.cast(image, dtype=tf.uint8)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

import time
start=time.time()
interpreter.invoke()

# Output is a [1, 1, 17, 3] numpy array.
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
print(time.time()-start)
img=cv2.imread(image_path)
h,w,c=img.shape
print(keypoints_with_scores)
for item in keypoints_with_scores[0][0]:
    #if item[2]>0.5:
    if item[2]>0.15:
        img = cv2.circle(img, (int(item[1]*w),int(item[0]*h)), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('out.png',img)

