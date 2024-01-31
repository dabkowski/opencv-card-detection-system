from threading import Thread

import cv2
import time
from ultralytics import YOLO
from ultralytics.solutions import object_counter

from constants import *

model = YOLO('model/best.pt')

im0 = None


def capture_frame(cap):
    global im0
    while cap.isOpened():
        success, im0 = cap.read()
        time.sleep(0.01)


def preprocess_card(counter, queue):
    global im0
    detected_ids = []
    detected_classes = []
    detected_defects = []
    counted_obj_l = []
    counted_obj = 0
    while True:
        tracks = model.track(im0, persist=True, conf=0.75, verbose=False)

        im0 = counter.start_counting(im0, tracks)

        if tracks[0].boxes.id is not None:
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for id in counter.counting_list:
                if id not in detected_ids:
                    for track_id, cls in zip(track_ids, clss):
                        if track_id == id:
                            counter.region_color = GREY_COLOR_BGR
                            detected_ids.append(id)
                            detected_classes.append(counter.names[cls])
                            counted_obj = counted_obj + 1
                            counted_obj_l.append(counted_obj)
                            break

        if counted_obj % 2 == 0 and counted_obj != 0:
            if detected_classes[counted_obj - 2] != detected_classes[counted_obj - 1]:
                counter.region_color = RED_COLOR_BGR
                CARD_VALUES[DEFECT_LABEL] = CARD_VALUES[DEFECT_LABEL] + 1
                detected_classes.pop(counted_obj - 1)
                detected_classes.pop(counted_obj - 2)
                counted_obj = counted_obj - 2
                detected_defects.append(DEFECT_LABEL)
            elif counted_obj_l[-1] != counted_obj_l[-2]:
                counter.region_color = GREEN_COLOR_BGR
                CARD_VALUES[detected_classes[-1]] = CARD_VALUES[detected_classes[-1]] + 1
                counted_obj_l.append(counted_obj)

        queue.put(CARD_VALUES)
        time.sleep(.04)


def detect_cards(queue, video_path):
    global im0

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # video_writer = cv2.VideoWriter("output_vids/object_counting_output.avi",
    #                                cv2.VideoWriter_fourcc(*'mp4v'),
    #                                fps,
    #                                (w, h))

    region_points = [(w - 140, h - 10), (w - 40, h - 10), (w - 40, 5), (w - 140, 5)]
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=region_points,
                     classes_names=model.names,
                     draw_tracks=True,
                     track_color=(124, 252, 0),
                     count_reg_color=GREY_COLOR_BGR,
                     view_in_counts=False,
                     view_out_counts=False)

    thread = Thread(target=capture_frame, args=(cap, fps))
    thread.daemon = True
    thread.start()

    thread2 = Thread(target=preprocess_card, args=(counter, queue))
    thread2.daemon = True
    thread2.start()

    thread.join()
    thread2.join()

    cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()
