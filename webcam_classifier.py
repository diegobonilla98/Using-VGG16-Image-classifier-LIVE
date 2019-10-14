import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from googletrans import Translator
import logging
from PIL import Image

logging.getLogger('tensorflow').disabled = True

translator = Translator()
model = VGG16()

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Capturing", img)
    img = Image.fromarray(img)

    img = img.resize((224, 224), Image.ANTIALIAS)
    image = img_to_array(img)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    label_es = translator.translate(label[1], dest='es', src='en').text
    print('%s en un %.2f%%' % (label_es.lower(), label[2] * 100))

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
