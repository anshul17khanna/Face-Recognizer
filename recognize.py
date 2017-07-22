import os
import sys
import cv2
import numpy as np
from shutil import copyfile

current_dir_path = os.path.dirname(os.path.abspath(__file__)) + '\\'

helpers = 'Helpers\\'

trained_dataset = 'Trained Dataset\\'

recognized_dataset = 'Recognized Dataset\\'

recognition_data = 'Recognition Data\\'
if len(sys.argv) == 2:
    recognition_data = sys.argv[1] + '\\'
if os.path.isdir(current_dir_path + recognition_data) is False:
    print "The directory \'{}\' does not exist in the current directory.\n\nTerminating the program .."\
        .format(recognition_data.strip().split('\\')[0])
    sys.exit()

face_cascade_path = helpers + 'haarcascade_frontalface_default.xml'


def trained_names():
    try:
        cur_data = os.listdir(trained_dataset)
    except:
        os.mkdir(trained_dataset)
        cur_data = os.listdir(trained_dataset)

    trained_dirs = []

    for item in cur_data:
        item_path = current_dir_path + trained_dataset + item

        if os.path.isdir(item_path):
            trained_dirs.append(item)

    return trained_dirs


def images_and_labels():
    dir_count = 0
    images = []
    labels = []

    all_data = os.listdir(trained_dataset)
    for current in all_data:
        if os.path.isdir(trained_dataset + current):
            dir_count += 1
            all_current_images = os.listdir(trained_dataset + current)

            for cur_img in all_current_images:
                labels.append(dir_count)
                img_path = trained_dataset + current + '\\' + cur_img

                img = cv2.imread(img_path, 1)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                images.append(gray_img)

    return images, labels


def train_recognizer():
    recognizer = cv2.face.createLBPHFaceRecognizer()

    images, labels = images_and_labels()
    if len(images) == 0:
        return False

    recognizer.train(np.array(images), np.array(labels))

    return recognizer


def recog(item, names):
    try:
        recognizer = train_recognizer()
    except:
        recognizer = False

    if recognizer is False:
        print "The trained dataset is empty!\n\nTerminating the program .."
        sys.exit()

    print 'Processing {} ..'.format(item)

    img_path = recognition_data + item
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX

    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    new_name = recognized_dataset + item.strip().split('.')[0]
    flag = False

    for (x, y, w, h) in faces:
        recognized_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        nbr, conf = recognizer.predict(gray_img[y: y + h, x: x + w])
        predicted_name = names[nbr - 1]
        if conf<90:
            print ' {} was recognized with confidence {}'.format(predicted_name, conf)
            new_name += '_' + predicted_name
            cv2.putText(recognized_img, predicted_name, (x, y - 5), font, 0.5, (255, 255, 255), 1)
        else:
            flag = True
            print ' Person was not recognized in the trained dataset.'
            cv2.putText(recognized_img, '[Unrecognized]', (x, y - 5), font, 0.5, (255, 255, 255), 1)

    if flag is True:
        new_name += '_Others'

    print '2 new files generated in {}\n'.format(current_dir_path + recognized_dataset)

    cv2.imwrite(recognized_dataset + item.strip().split('.')[0]+'_recognized'+'.'+item.strip().split('.')[1], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    copyfile(img_path, new_name + '.' + item.strip().split('.')[1])

    return


def main():
    recog_items = os.listdir(recognition_data)
    names = trained_names()

    for item in recog_items:
        if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
            recog(item, names)
        else:
            print '{} format not supported. Can\'t recognize {} ..\n'.format(item.strip().split('.')[1], item)

    return

if __name__ == "__main__":
    main()
