from step0_get_page_name import doc3D
import cv2 
import numpy as np 

# print(doc3D.page_names)
# print(doc3D.db_root + "/" + "wc" + "/" + doc3D.page_names[0] + ".exr")
import os 

names = []
# for i in range(1, 22):
for i in [1, 10, 11]:
    names += ["D:/swat3D/wc/%i/" % i + name for name in os.listdir("D:/swat3D/wc/%i" % i)]

print(names)

# wcs = []
# for i, name in enumerate(names[:10]):
#     print(i, name)
#     wcs.append(cv2.imread(name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))  ### 這行就可以了！

# # for i, page_name in enumerate(doc3D.page_names[:]):
# #     wc_path = doc3D.db_root + "/" + "wc" + "/" + page_name + ".exr"
# #     print(i, wc_path)
# #     wcs.append(cv2.imread(wc_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))  ### 這行就可以了！

# # 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497 # RGB -> BGR

# wcs  = np.array(wcs)
# print(wcs.shape)
# print("ch0 max:", wcs[..., 0].max())
# print("ch0 min:", wcs[..., 0].min())
# print("ch1 max:", wcs[..., 1].max())
# print("ch1 min:", wcs[..., 1].min())
# print("ch2 max:", wcs[..., 2].max())
# print("ch2 min:", wcs[..., 2].min())
