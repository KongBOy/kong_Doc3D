# from step0_get_page_name import doc3D
import sys
sys.path.append(r"C:\Users\VAIO_Kong\Desktop\kong_model2\kong_util")
from util import multi_processing_interface
from multiprocessing import Manager
import cv2
import numpy as np
from tqdm import tqdm
import os
import time


def wx_xyz_min_max_multiprocess(wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins, core_amount, task_amount):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=wc_xyz_min_max, task_args=[wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins], print_msg=True)

def wc_xyz_min_max(start_index, amount, wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins):
    wcs = []
    # start_index = 0
    # amount = 100
    for i, file_path in enumerate(tqdm(wc_paths[start_index:start_index + amount])):
        # print(i, file_path)
        wcs.append(cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))  ### 這行就可以了！

    wcs  = np.array(wcs)
    # print(wcs.shape)
    # print("ch0 max:", wcs[..., 0].max())
    # print("ch0 min:", wcs[..., 0].min())
    # print("ch1 max:", wcs[..., 1].max())
    # print("ch1 min:", wcs[..., 1].min())
    # print("ch2 max:", wcs[..., 2].max())
    # print("ch2 min:", wcs[..., 2].min())
    # print()
    ch0_maxs.append( wcs[..., 0].max())
    ch0_mins.append( wcs[..., 0].min())
    ch1_maxs.append( wcs[..., 1].max())
    ch1_mins.append( wcs[..., 1].min())
    ch2_maxs.append( wcs[..., 2].max())
    ch2_mins.append( wcs[..., 2].min())


if(__name__ == "__main__"):
    wc_root = r"D:\swat3D\wc"
    # wc_root = "wc"
    wc_paths = []
    ### 取得 wc_paths
    for i in range(1, 22):
        wc_paths += [ wc_root + "/%i/" % i + wc_name for wc_name in os.listdir( wc_root + "/%i" % i)]
        ### 理想 wc_path = doc3D.db_root + "/" + "wc" + "/" + page_name + ".exr"，有空再改

    ### 設定 要處理的數量
    # wc_amount = 2000 #len(wc_paths)
    wc_amount = len(wc_paths)
    # for wc_path in wc_paths: print(wc_path)
    print("wc_amount:", wc_amount)

    ############################################################################################################
    start_time = time.time()
    with Manager() as manager:  ### 設定在 multiprocess 裡面 共用的 list
        ### 以下想 multiprocess
        ### global 的 list，應該就要用 share memory 了
        ch0_maxs = manager.list()  # []
        ch0_mins = manager.list()  # []
        ch1_maxs = manager.list()  # []
        ch1_mins = manager.list()  # []
        ch2_maxs = manager.list()  # []
        ch2_mins = manager.list()  # []
        core_amount = wc_amount // 150
        task_amount = wc_amount
        wx_xyz_min_max_multiprocess(wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins, core_amount=core_amount, task_amount=task_amount)

        print("ch0_maxs:", ch0_maxs)
        print("ch0_mins:", ch0_mins)
        print("ch1_maxs:", ch1_maxs)
        print("ch1_mins:", ch1_mins)
        print("ch2_maxs:", ch2_maxs)
        print("ch2_mins:", ch2_mins)
        print()

        ch0_maxs = np.array(ch0_maxs)
        ch0_mins = np.array(ch0_mins)
        ch1_maxs = np.array(ch1_maxs)
        ch1_mins = np.array(ch1_mins)
        ch2_maxs = np.array(ch2_maxs)
        ch2_mins = np.array(ch2_mins)

        print("ch0_maxs.max()", ch0_maxs.max())
        print("ch0_mins.min()", ch0_mins.min())
        print("ch1_maxs.max()", ch1_maxs.max())
        print("ch1_mins.min()", ch1_mins.min())
        print("ch2_maxs.max()", ch2_maxs.max())
        print("ch2_mins.min()", ch2_mins.min())

    print("total_cost_time:", time.time() - start_time)
    """
    ch0_maxs.max() 0.63452387
    ch0_mins.min() -0.67187124
    ch1_maxs.max() 1.2387834
    ch1_mins.min() -1.2280148
    ch2_maxs.max() 1.2485291
    ch2_mins.min() -1.2410645
    total_cost_time: 4970.9831392765045
    total_cost_time: 4822.389922380447
    """

# # for i, page_name in enumerate(doc3D.page_names[:]):
# #     wc_path = doc3D.db_root + "/" + "wc" + "/" + page_name + ".exr"
# #     print(i, wc_path)
# #     wcs.append(cv2.imread(wc_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))  ### 這行就可以了！

# # 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497 # RGB -> BGR
