import numpy as np
import cv2
from FeatureExtractor import Extractor
# import time

fe = Extractor()


def process_frame(img):
    # print(img.shape)
    # cv2.resize(img, ())
    matches = fe.extract(img)
    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        img = cv2.circle(img, (u1, v1), 1, (0, 255, 0), 1)
        img = cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0), 1)
    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture('data/test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        frame = process_frame(frame)
        cv2.imshow('test.mpt', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()