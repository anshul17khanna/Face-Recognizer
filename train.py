import os
import cv2
from Tkinter import *
from PIL import Image, ImageTk
from shutil import copyfile
import random

current_dir_path = os.path.dirname(os.path.abspath(__file__)) + '\\'

helpers = 'Helpers\\'

trained_dataset = 'Trained Dataset\\'

training_data = 'Training Data\\'
if len(sys.argv) == 2:
    training_data = sys.argv[1] + '\\'
if os.path.isdir(current_dir_path + training_data) is False:
    print "The folder \'{}\' does not exist in the current directory.\n\nTerminating the program .."\
        .format(training_data.strip().split('\\')[0])
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


def rename_files(trained_dirs):
    for dir in trained_dirs:
        dir_path = trained_dataset + dir + '\\'
        dir_items = os.listdir(dir_path)

        for item in dir_items:
            if item.startswith(dir + '_') is False or re.search(r'\d+$', item.strip().split('.')[0]) is None:
                new_name = dir + '_' + str(random.randint(1, 999999)) + '.' + item.strip().split('.')[1]
                os.rename(dir_path + item, dir_path + new_name)

    return


def train(item, names):
    print 'Processing {} ..'.format(item)

    img_path = training_data + item
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_img = img[y:y+w, x:x+h]
        cv2.imwrite(training_data + 'tmp.jpg', face_img)

        root = Tk()

        window_img = ImageTk.PhotoImage(Image.open(training_data + 'tmp.jpg'))

        window_img_label = Label(root, image=window_img)
        window_img_label.pack()

        frame = Frame(root)
        frame.grid()

        def callback(txt):
            if txt != '[discard]':
                new_path = trained_dataset + txt + '\\'

                copyfile(training_data + 'tmp.jpg', new_path + 'tmp' + '.' + item.strip().split('.')[1])
                print 'Person trained : {}'.format(txt)

            root.destroy()

        question_label = Label(root, text='Who is this ?')
        question_label.pack()

        i = j = 0
        for each_name in names:
            name_button = Button(frame, text=each_name, command=lambda j=each_name: callback(j))
            name_button.config(bd=4, width=12)
            name_button.grid(row=i, column=j)

            j += 1
            j %= 4
            if j == 0:
                i += 1

        name_button = Button(frame, text='[discard]', command=lambda j='[discard]': callback(j))
        name_button.config(bd=4, width=12)
        name_button.grid(row=i, column=j)

        frame.pack()

        add_name_label = Label(root, text='Add a new name :')
        add_name_label.pack()

        frame = Frame(root)
        frame.grid()

        def new_name(add_txt):
            txt = add_txt.get()
            new_path = trained_dataset + txt + '\\'

            if os.path.isdir(new_path) is False:
                os.mkdir(new_path)
                print 'New person added : {}'.format(txt)

            copyfile(training_data + 'tmp.jpg', new_path + 'tmp' + '.' + item.strip().split('.')[1])
            print 'Person trained : {}'.format(txt)

            names.append(txt)

            root.destroy()

        entry = Entry(root, width=20)
        entry.pack()

        add_button = Button(frame, text='Add', command=lambda j=entry: new_name(j))
        add_button.config(bd=4)
        add_button.grid(row=0)

        frame.pack()

        empty_label = Label(root, text='')
        empty_label.pack()

        mainloop()

        os.remove(training_data + 'tmp.jpg')

    print
    rename_files(names)

    return


def main():
    training_items = os.listdir(training_data)
    names = trained_names()

    for item in training_items:
        if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
            train(item, names)
        else:
            print '{} format not supported. Can\'t train {} ..\n'.format(item.strip().split('.')[1], item)

    return

if __name__ == "__main__":
    main()
