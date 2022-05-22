'''
augtexnames.txt 這應該是給 DewarpNet 用的喔～
'''
import os
from step0_Doc3D_obj import real_doc3D

aug_path = real_doc3D.db_root[: len(real_doc3D.db_root.split("/")[-1]) * -1 - 1] + "/" + "augtexnames.txt"
if os.path.isfile(aug_path): os.remove(aug_path)

dtd_img_root_read_dir = r"J:\dtd\images"
dtd_img_root_write_str = "/dtd/images"  ### 最前面一訂要加 / 喔！ 為了 配合 人家寫的code這樣子拉
dtd_img_dirs = os.listdir(dtd_img_root_read_dir)
names = []
for img_dir in dtd_img_dirs:
    names += [dtd_img_root_write_str + "/" + img_dir + "/" + name for name in os.listdir(dtd_img_root_read_dir + "/" + img_dir) if ".jpg" in name]

with open(aug_path, "w") as f:
    for name in names:
        f.write(name + "\n")
