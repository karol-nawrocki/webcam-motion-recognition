import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def get_image_from_camera(camera: cv2.VideoCapture, resize=None, grayscale=False):
    _, frame = camera.read()
    if resize:
        array_size = (int(frame.shape[1] * resize), int(frame.shape[0] * resize))
        frame = cv2.resize(frame, dsize=array_size, interpolation=cv2.INTER_CUBIC)
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def get_diff_between_frames(prev_frame, curr_frame):
    image_size = curr_frame.shape[0] * curr_frame.shape[1]
    return np.sum(np.absolute(np.subtract(prev_frame.astype('int8'), curr_frame.astype('int8')))) / image_size


def get_available_cameras_indexes():
    indexes = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            indexes.append(i)
            cap.release()
    return indexes


IMAGE_RESIZE = False
USE_GRAYSCALE = True
PIXEL_DIFF_THRESHOLD = 3
INITIALIZATION_OFFSET = 10
OUTPUT_DIR = Path(r'output')
SHOW_PLOTS = True


if __name__ == '__main__':
    cameras = tuple({'idx': idx,
                     'video_capture': cv2.VideoCapture(idx),
                     'previous_frame': get_image_from_camera(cv2.VideoCapture(idx), grayscale=USE_GRAYSCALE, resize=IMAGE_RESIZE),
                     'diffs': [],
                     'exceeded_threshold_x': [],
                     'exceeded_threshold_y': []
                     } for idx in get_available_cameras_indexes()
                    )
    plt.gray()
    frame_number = 0

    print('Initialization complete!')
    while True:
        frame_number += 1
        for i, cam in enumerate(cameras):
            cam['current_frame'] = get_image_from_camera(cam['video_capture'], grayscale=USE_GRAYSCALE, resize=IMAGE_RESIZE)
            avg_diff = get_diff_between_frames(cam['previous_frame'], cam['current_frame'])
            cam['diffs'].append(avg_diff)
            cam['previous_frame'] = cam['current_frame']
            if avg_diff > PIXEL_DIFF_THRESHOLD and frame_number > INITIALIZATION_OFFSET:
                print(f'Threshold exceeded: {avg_diff} (frame {frame_number})')
                timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")
                file_name = f'camera_{cam["idx"]}_{timestamp}_frame_{frame_number}.png'
                cam['exceeded_threshold_x'].append(frame_number - 1)
                cam['exceeded_threshold_y'].append(avg_diff)
                cv2.imwrite(str(OUTPUT_DIR / file_name), cam['current_frame'])
                cv2.imwrite(str(OUTPUT_DIR / file_name).replace('png', 'jpg'), cam['current_frame'])
            if SHOW_PLOTS:
                plt.subplot(len(cameras), len(cameras), i + 1)
                plt.plot(range(len(cam['diffs'])), cam['diffs'])
                plt.scatter(cam['exceeded_threshold_x'], cam['exceeded_threshold_y'], c='red', s=5)
                plt.subplot(len(cameras), len(cameras), i + 1 + len(cameras))
                plt.imshow(cam['current_frame'], interpolation='nearest')

        if SHOW_PLOTS:
            plt.pause(0.01)
            plt.clf()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        print(f'Releasing camera {cam["idx"]}')
        cam['video_capture'].release()
