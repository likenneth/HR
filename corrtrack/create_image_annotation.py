import os
import json
from tqdm import tqdm

# fg_dir = "/private/home/keli22/datasets/FineGym"
# ipt_dir = os.path.join(fg_dir, "processed_frames")
# opt_dir = os.path.join(fg_dir, "posetrack_data")

# for video_name in tqdm(os.listdir(ipt_dir)):
#     tbd = {"images": [], }
#     length = len(os.listdir(os.path.join(ipt_dir, video_name)))
#     for frame_jpg in os.listdir(os.path.join(ipt_dir, video_name)):
#         fidx = int(frame_jpg.split(".")[0])
#         tba = {
#             "file_name": f"processed_frames/{video_name}/{frame_jpg}", 
#             "has_labeled_person": True, 
#             "vid_id": video_name, 
#             "id": f"{video_name}_{fidx:03}", 
#             "image_id": f"{video_name}_{fidx:03}", 
#             "is_labeled": False, 
#             "nframes": length, 
#         }
#         tbd["images"].append(tba)
#     with open(os.path.join(opt_dir, f"{video_name}.json"), "w") as f:
#         json.dump(tbd, f)

fg_dir = "/private/home/keli22/datasets/PoseTrack21"
ipt_dir = os.path.join(fg_dir, "images/test")
opt_dir = os.path.join(fg_dir, "posetrack_data/test")

for video_name in tqdm(os.listdir(ipt_dir)):
    tbd = {"images": [], }
    video_index = video_name.split("_")[0]
    length = len(os.listdir(os.path.join(ipt_dir, video_name)))
    for frame_jpg in os.listdir(os.path.join(ipt_dir, video_name)):
        fidx = int(frame_jpg.split(".")[0])
        tba = {
            "file_name": f"images/test/{video_name}/{frame_jpg}", 
            "has_labeled_person": True, 
            "vid_id": video_name, 
            "id": int(f"1{video_index}{fidx:04}"), 
            "image_id": int(f"1{video_index}{fidx:04}"), 
            "is_labeled": False, 
            "nframes": length, 
        }
        tbd["images"].append(tba)
    with open(os.path.join(opt_dir, f"{video_name}.json"), "w") as f:
        json.dump(tbd, f)
