import os
import sys
import math

def save_tasks(range_x, range_y, orientation, out_file):
    for pos_x, pos_y in zip(range_x, range_y):
        out_file.write("{0},{1},{2}\n".format(pos_x, pos_y, orientation))


def get_file_path():
    target_file = os.path.join(os.path.join(os.getcwd(), "config"), "tracking_tasks.tmp.txt")
    return target_file

def write_diamond_track(target_file):
    with open(target_file, 'w') as tf:
        # Orientation in anti-clockwise degree
        orientation_xy = 45
        x_range = range(0, -100, -10)
        y_range = range(0, -100, -10)
        save_tasks(x_range, y_range, orientation_xy, tf)

        orientation_xy = 135
        x_range = range(-100, -200, -10)
        y_range = range(-100, 0, 10)
        save_tasks(x_range, y_range, orientation_xy, tf)

        orientation_xy = 225
        x_range = range(-200, -100, 10)
        y_range = range(0, 100, 10)
        save_tasks(x_range, y_range, orientation_xy, tf)

        orientation_xy = 315
        x_range = range(-100, 0, 10)
        y_range = range(100, 0, -10)
        save_tasks(x_range, y_range, orientation_xy, tf)

def write_cicle_track(target_file="tracking_tasks.tmp.txt", circle_x0=0, circle_y0=0, radius=120, steps=36):
    with open(target_file, 'w') as tf:
        for step in range(steps + 1):
            ort_angle = int(360 / steps * step)
            pos_diff_x = abs(int(radius * math.cos(ort_angle / 180 * math.pi)))
            pos_diff_y = abs(int(radius * math.sin(ort_angle / 180 * math.pi)))

            x_ind = 1
            y_ind = 1

            if 0 <= ort_angle <= 90:
                x_ind = 1
                y_ind = -1
            elif 90 < ort_angle <= 180:
                x_ind = -1
                y_ind = -1
            elif 180 < ort_angle <= 270:
                x_ind = -1
                y_ind = 1
            elif 270 < ort_angle <= 360:
                x_ind = 1
                y_ind = 1
            else:
                x_ind = 1
                y_ind = 1

            pos_x = pos_diff_x * x_ind + circle_x0
            pos_y = pos_diff_y * y_ind + circle_y0

            tf.write("{0:d},{1:d},{2:d}\n".format(pos_x, pos_y, ort_angle))


tmp_filepath = get_file_path()
write_cicle_track(target_file=tmp_filepath, circle_x0=-300, circle_y0=0, radius=300, steps=36)

