# Import TF and TF Hub libraries.
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Load the input image.
image_path = './sample.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

import time
start=time.time()
# Run model inference.
outputs = movenet(image)
print(time.time()-start)
img=cv2.imread(image_path)
h,w,c=img.shape
# Output is a [1, 1, 17, 3] tensor.
keypoints = outputs['output_0']
print(keypoints)

for item in keypoints[0][0]:
    if item[2]>0.5:
        img = cv2.circle(img, (int(item[1]*w),int(item[0]*h)), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('out.png',img)

