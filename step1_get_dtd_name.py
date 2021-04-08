import os
from step0_get_page_name import doc3D

aug_path = doc3D.db_root[: len(doc3D.db_root.split("/")[-1]) * -1 - 1] + "/" + "augtexnames.txt"
if os.path.isfile(aug_path): os.remove(aug_path)

dtd_img_root = "dtd/images"
dtd_img_dirs = os.listdir(dtd_img_root)
names = []
for img_dir in dtd_img_dirs:
    names += [dtd_img_root + "/" + img_dir + "/" + name for name in os.listdir(dtd_img_root + "/" + img_dir)]

with open(aug_path, "w") as f:
    for name in names:
        f.write(name + "\n")
