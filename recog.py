
def encode(path):
    directory = path
    encodes = []
    names = []
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        x = filename.split(".")[0]
        names.append(x)
        # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f)
            encodes.append(face_recognition.face_encodings(img)[0])
    return [names,encodes]

def recognize(label, path, data):
    name= label
    img= cv2.imread(path)
    img_encoding = face_recognition.face_encodings(img)
    if img_encoding != []:
            img_encoding = img_encoding[0]
            result= face_recognition.compare_faces(data[1],img_encoding)
            if True in result:
                x= result.index(True)
                name= data[0][x]
                return name
    else:
        return name

def extract(image,left, top, right, bottom):
    plt.ioff()
    fig = plt.figure(figsize=(6.4,6.4))
    ax1 = plt.subplot(1, 1, 1)
    plt.imshow(image[top:bottom, left:right])
    fig.add_subplot(ax1)
    fig.savefig("/content/drive/MyDrive/dataset/temp/x.jpg")
    plt.close(fig)
    path = "/content/drive/MyDrive/dataset/temp/x.jpg"
    return path


def mainFunc(imagePath):
    image = cv2.imread(imagePath)
    detections, width_ratio, height_ratio = darknet_helper(image, width, height)
    for label, confidence, bbox in detections:
        if (float(confidence)<65):
            continue
        elif ((label=="person") and (float(confidence)<85)):
            continue
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        img = extract(image, left, top, right, bottom)
        name = recognize(label, img, data)
        if name == None:
            name = "person"
        print("Detected: ",name, " ")
        print("Confidence: ",confidence, "\n")
        cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(image, "{} [{:.2f}]".format(name, float(confidence)),
                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            class_colors[label], 2)
        
    cv2.imshow(image)


    