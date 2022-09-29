import glob, os

#Current directory
#current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)

current_dir = 'C:/Users/zuhal/Desktop/termal_drone/yolov4/darknet/data/obj'

percentage_test = 10

file_train = open('C:/Users/zuhal/Desktop/termal_drone/yolov4/darknet/data/train.txt', 'w')
file_test = open('C:/Users/zuhal/Desktop/termal_drone/yolov4/darknet/data/test.txt', 'w')
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
    else:
        file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        counter = counter + 1