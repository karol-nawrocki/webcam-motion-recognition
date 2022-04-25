import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def get_image_from_camera(camera: cv2.VideoCapture, resize=None):
    _, frame = camera.read()
    if resize:
        array_size = (int(frame.shape[1] * resize), int(frame.shape[0] * resize))
        frame = cv2.resize(frame, dsize=array_size, interpolation=cv2.INTER_CUBIC)
    return frame


def get_diff_between_frames(prev_frame, curr_frame):
    image_size = curr_frame.shape[0] * curr_frame.shape[1]
    frame1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    return np.sum(np.absolute(np.subtract(frame1.astype('int8'), frame2.astype('int8')))) / image_size


def get_available_cameras_indexes():
    indexes = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            indexes.append(idx)
            cap.release()
    return indexes


IMAGE_RESIZE = False
PIXEL_DIFF_THRESHOLD = 10
INITIALIZATION_OFFSET = 20
PLOT_SAVE_INTERVAL = 2500
OUTPUT_DIR = Path(r'output')
SHOW_PLOTS = True


if __name__ == '__main__':
    cameras = tuple({'idx': idx,
                     'video_capture': cv2.VideoCapture(idx),
                     'previous_frame': get_image_from_camera(cv2.VideoCapture(idx), resize=IMAGE_RESIZE),
                     'diffs': [],
                     'exceeded_threshold_x': [],
                     'exceeded_threshold_y': []
                     } for idx in get_available_cameras_indexes()
                    )
    frame_number = 0

    plt.figure(figsize=(12, 6), dpi=80)

    print('Initialization complete!')
    while True:
        for cam_idx, cam in enumerate(cameras):
            cam['current_frame'] = get_image_from_camera(cam['video_capture'], resize=IMAGE_RESIZE)
            avg_diff = get_diff_between_frames(cam['previous_frame'], cam['current_frame'])
            cam['diffs'].append(avg_diff)
            cam['previous_frame'] = cam['current_frame']
            timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")
            if avg_diff > PIXEL_DIFF_THRESHOLD and frame_number > INITIALIZATION_OFFSET:
                print(f'Threshold exceeded: {avg_diff} (frame {frame_number})')
                file_name = f'camera_{cam["idx"]}_{timestamp}_frame_{frame_number}.jpg'
                cam['exceeded_threshold_x'].append(frame_number)
                cam['exceeded_threshold_y'].append(avg_diff)
                cv2.imwrite(str(OUTPUT_DIR / file_name), cam['current_frame'])
            if SHOW_PLOTS:
                plt.subplot(len(cameras), 2, (cam_idx * 2) + 1)
                plt.plot(range(len(cam['diffs'])), cam['diffs'])
                plt.scatter(cam['exceeded_threshold_x'], cam['exceeded_threshold_y'], c='red', s=5)
                plt.subplot(len(cameras), 2, (cam_idx * 2) + 2)
                plt.imshow(cam['current_frame'], interpolation='nearest')

                # Save only when it's the last camera, so all plots are included in the image:
                if frame_number % PLOT_SAVE_INTERVAL == 0 and cam['idx'] == [c['idx'] for c in cameras][-1]:
                    plt.savefig(f'plot_{timestamp}_frame_{frame_number}')

        if SHOW_PLOTS:
            plt.pause(0.01)
            plt.clf()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_number += 1

    for cam in cameras:
        print(f'Releasing camera {cam["idx"]}')
        cam['video_capture'].release()
