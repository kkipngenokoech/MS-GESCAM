from GESCAMDATA.videoconstructor import reconstruct_video
import os
import cv2

output_dir = "videos"

paths = [
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/test_subset/task_classroom_11_video-01_final/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video01(301-600)/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video02(0-300)/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video02(301-600)/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video03_final/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video04_final/images",
    "/home/kip/projects/MS-GESCAM/GESCAMDATA/train_subset/Classroom 01/task_classroom_01_video05_final/images"
]


for path in paths:
    reconstruct_video(
        path,
        output_video_dir=output_dir,
        frame_rate=30
    )