from glob import glob 
import tqdm
import os

rgb_root = "dataset/GaitLU_img/train/org"
pose_root = "dataset/GaitLU_img/train/pose"
sil_root = "dataset/GaitLU_img/train/sil"
city_name="*"

all_ids_list_rgb = sorted(glob(os.path.join(rgb_root, f"{city_name}/*")))
all_ids_list_pose = sorted(glob(os.path.join(pose_root, f"{city_name}/*")))
all_ids_list_sil = sorted(glob(os.path.join(sil_root, f"{city_name}/*")))

print(len(all_ids_list_rgb), len(all_ids_list_pose), len(all_ids_list_sil))


list_rgb = []
list_pose = []
list_sil = []

for i in range(len(all_ids_list_rgb)):
    rgb_city_id_name = '-'.join(all_ids_list_rgb[i].split("/")[-2:])
    list_rgb.append(rgb_city_id_name)

for i in range(len(all_ids_list_pose)):
    pose_city_id_name = '-'.join(all_ids_list_pose[i].split("/")[-2:])
    list_pose.append(pose_city_id_name)

for i in range(len(all_ids_list_sil)):
    sil_city_id_name = '-'.join(all_ids_list_sil[i].split("/")[-2:])
    list_sil.append(sil_city_id_name)

print(len(list_rgb), len(list_pose), len(list_sil))


# for rgb in list_rgb:
#     if rgb not in list_pose:
#         different_city = rgb.split("-")[0]
#         different_id = rgb.split("-")[1]
#         different_path_rgb = os.path.join(rgb_root, different_city, different_id)
#         different_path_sil = os.path.join(sil_root, different_city, different_id)

#         os.system(f"rm -rf {different_path_rgb}")
#         os.system(f"rm -rf {different_path_sil}")


