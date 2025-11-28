from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro

from gello.zmq_core.camera_node import ZMQClientCamera


@dataclass
class Args:
    # For multiple cameras
    ports: Tuple[int, ...] = (4000, 4001)
    hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"

    # For single camera
    # port = 4000

def main(args):
    cameras = []
    import cv2

    images_display_names = []
    for port in args.ports:
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(f"image_{port}")
        cv2.namedWindow(images_display_names[-1], cv2.WINDOW_NORMAL)

    # For single camera
    # cameras.append(ZMQClientCamera(port=args.port, host=args.hostname))
    # images_display_names.append(f"image_{args.port}")
    while True:
        for display_name, camera in zip(images_display_names, cameras):
            image, depth = camera.read()
            # stacked_depth = np.dstack([depth, depth, depth]).astype(np.uint8)
            # image_depth = cv2.hconcat([image[:, :, ::-1], stacked_depth])
            cv2.imshow(display_name, image)
            cv2.waitKey(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
