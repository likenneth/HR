# will create a folder called posetrack_data in the dataset folder, that serve as some meta data for corrtrack code base

from genericpath import exists
import os
import json
from tqdm import tqdm

#######################################################################################################
# for finegym at action level # but some of the temporal boundaries are out of the length of the videos
#######################################################################################################

# fg_dir = "/private/home/keli22/datasets/FineGym"
# ipt_dir = os.path.join(fg_dir, "processed_frames")
# opt_dir = os.path.join(fg_dir, "posetrack_data")
# fps = 25
# def get_annos():
#     with open("/private/home/keli22/datasets/FineGym/finegym_annotation_info_v1.1.json", "r") as f:
#         gym_annot = json.load(f)
#     action_list = []
#     for video, events in gym_annot.items():
#         for event, event_details in events.items():
#             if event_details["segments"]:
#                 for action, action_details in event_details["segments"].items():
#                     relative_time_stamp = [action_details["timestamps"][0][0], action_details["timestamps"][-1][-1]]  # relative to events
#                     s = round(relative_time_stamp[0] * fps)
#                     e = round(relative_time_stamp[1] * fps)
#                     if e <= s:
#                         continue
#                     action_list.append(
#                         {
#                             "event_frames": os.path.join(fg_dir, "processed_frames", f"{video}_{event}"), 
#                             "time_stamp": relative_time_stamp, 
#                             "action_name": f"{video}_{event}_{action}", 
#                             "v": video, 
#                             "e": event, 
#                             "a": action, 
#                         }
#                     )

#     return action_list

# anno = get_annos()
# print(f"Total Finegym actions {len(anno)}")
# for item in tqdm(anno):
#     action_name = item["action_name"]
#     video, event, action = item["v"], item["e"], item["a"]
#     relative_time_stamp = item["time_stamp"]

#     s = round(relative_time_stamp[0] * fps)
#     e = round(relative_time_stamp[1] * fps)

#     tbd = {"images": [], }
#     length = e - s + 1
#     for fidx in range(s, e+1):
#         tba = {
#             "file_name": f"processed_frames/{video}_{event}/{fidx+1:03}.jpg", 
#             "has_labeled_person": True, 
#             "vid_id": action_name, 
#             "id": f"{action_name}_{fidx+1:03}", 
#             "image_id": f"{action_name}_{fidx+1:03}", 
#             "is_labeled": False, 
#             "nframes": length, 
#         }
#         tbd["images"].append(tba)
#         assert os.path.exists(os.path.join(fg_dir, tba["file_name"])), print(f"{fg_dir} total {s}--{e}")
#     with open(os.path.join(opt_dir, f"{action_name}.json"), "w") as f:
#         json.dump(tbd, f)

########################
# for posetrack21 test #
########################

# fg_dir = "/private/home/keli22/datasets/PoseTrack21"
# ipt_dir = os.path.join(fg_dir, "images/test")
# opt_dir = os.path.join(fg_dir, "posetrack_data/test")

# for video_name in tqdm(os.listdir(ipt_dir)):
#     tbd = {"images": [], }
#     video_index = video_name.split("_")[0]
#     length = len(os.listdir(os.path.join(ipt_dir, video_name)))
#     for frame_jpg in os.listdir(os.path.join(ipt_dir, video_name)):
#         fidx = int(frame_jpg.split(".")[0])
#         tba = {
#             "file_name": f"images/test/{video_name}/{frame_jpg}", 
#             "has_labeled_person": True, 
#             "vid_id": video_name, 
#             "id": int(f"1{video_index}{fidx:04}"), 
#             "image_id": int(f"1{video_index}{fidx:04}"), 
#             "is_labeled": False, 
#             "nframes": length, 
#         }
#         tbd["images"].append(tba)
#     with open(os.path.join(opt_dir, f"{video_name}.json"), "w") as f:
#         json.dump(tbd, f)


###########################
# for finegym event level #
###########################

fg_dir = "/private/home/keli22/datasets/FineGym"
ipt_dir = os.path.join(fg_dir, "processed_frames")
opt_dir = os.path.join(fg_dir, "posetrack_data")
os.makedirs(opt_dir, exist_ok=True)

for video_name in tqdm(os.listdir(ipt_dir)):
    tbd = {"images": [], }
    video_index = video_name.split("_")[0]
    length = len(os.listdir(os.path.join(ipt_dir, video_name)))
    for frame_jpg in sorted(list(os.listdir(os.path.join(ipt_dir, video_name)))):
        fidx = int(frame_jpg.split(".")[0])  # starting from 1
        tba = {
            "file_name": f"processed_frames/{video_name}/{frame_jpg}", 
            "has_labeled_person": True, 
            "vid_id": video_name, 
            "id": f"{video_name}_{fidx:03}", 
            "image_id": f"{video_name}_{fidx:03}", 
            "is_labeled": False, 
            "nframes": length, 
        }
        tbd["images"].append(tba)
    with open(os.path.join(opt_dir, f"{video_name}.json"), "w") as f:
        json.dump(tbd, f)
