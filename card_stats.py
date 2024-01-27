import numpy as np
import cv2
from constants import *

canvas_width = 880
canvas_height = 280


def create_canvas(height, width, color_bgr):
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:, :, :] = color_bgr
    return blank_image


def difference_between_dicts(dict1, dict2):
    different_values = {key: (dict1[key], dict2[key]) for key in set(dict1) & set(dict2) if dict1[key] != dict2[key]}

    different_values.update({key: (dict1[key], None) for key in set(dict1) - set(dict2)})
    different_values.update({key: (None, dict2[key]) for key in set(dict2) - set(dict1)})
    return different_values


def view_card_count_window(queue):
    previous_card_values = None

    while True:
        blank_canvas = create_canvas(canvas_height, canvas_width, WHITE_COLOR_BGR)
        new_dict = queue.get()

        difference = None
        if previous_card_values is not None:
            difference = difference_between_dicts(previous_card_values, new_dict)

        start_x_pos = 30
        start_y_pos = -200
        loop_counter = 0
        for key, val in queue.get().items():
            font_color = LIGHT_FONT_BLACK
            if difference is not None and key in difference.keys() and key == DEFECT_LABEL:
                font_color = RED_COLOR_BGR
            elif difference is not None and key in difference.keys():
                font_color = GREEN_COLOR_BGR

            if loop_counter % 13 == 0:
                start_x_pos = 30
                start_y_pos = start_y_pos + 200
            loop_counter = loop_counter + 1
            cv2.putText(blank_canvas, key + ": " + str(val), (start_y_pos, start_x_pos), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        font_color, 1, cv2.LINE_AA)
            start_x_pos = start_x_pos + 20

        previous_card_values = new_dict
        cv2.imshow("Cards count", blank_canvas)
        cv2.waitKey(1)
