import cv2 as cv
import numpy as np
import dlib

meme_man = cv.imread('meme_man.png', cv.IMREAD_UNCHANGED)
meme_man = cv.cvtColor(meme_man, cv.COLOR_BGR2BGRA)

# Resize Meme Man
meme_man = cv.resize(meme_man, (370, 370), interpolation=cv.INTER_AREA)

xOffset = -90
yOffset = -160

# Make empty meme man alpha image
meme_man_alpha = np.zeros((meme_man.shape[0], meme_man.shape[1], 4), dtype='uint8')

meme_man_height, meme_man_width, meme_man_c = meme_man.shape
for i in range(0, meme_man_height):
    for j in range(0, meme_man_width):
        if meme_man[i, j][3] != 0:
            meme_man_alpha[i, j] = meme_man[i, j]

b, g, r, a = cv.split(meme_man_alpha)


capture = cv.VideoCapture(0)

# Detects faces
detector = dlib.get_frontal_face_detector()

# Detects face landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = capture.read()

    frame_meme_man = np.zeros((frame.shape[0], frame.shape[1], 4), dtype='uint8')

    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

    # Detect faces
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    # For all detected faces
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Detect Landmark Points on a face
        landmarks = predictor(gray, face)

        # Place circles
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(frame, (x, y), 4, (255, 0, 0), -1)
        x = landmarks.part(0).x + xOffset
        y = landmarks.part(0).y + yOffset
        print(landmarks.part(0))
        frame_meme_man = np.zeros((frame.shape[0], frame.shape[1], 4), dtype='uint8')
        _, mask = cv.threshold(a, 0, 255, cv.THRESH_BINARY)
        try:
            mask_full_size = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
            mask_full_size[y:mask.shape[1] + y, x:mask.shape[0] + x] = mask[:, :]

            # Invert mask_full_size
            mask_full_size = cv.bitwise_not(mask_full_size)

            frame = cv.bitwise_and(frame, frame, mask=mask_full_size)

            frame_meme_man[y:meme_man_height + y, x:meme_man_width + x, 0] = b
            frame_meme_man[y:meme_man_height + y, x:meme_man_width + x, 1] = g
            frame_meme_man[y:meme_man_height + y, x:meme_man_width + x, 2] = r
        except:
            pass

    cv.imshow('Output', frame_meme_man+frame)

    key = cv.waitKey(1)
    if key == 27:
        break
