#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
sys.path.append(kong_model2_dir + "/kong_util")
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
from kong_util.util import get_exr
from kong_util.multiprocess_util import multi_processing_interface
from kong_util.wc_util import wc_3d_plot, wc_2d_plot, uv_2d_plot
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Save_as_jpg, Save_npy_path_as_knpy
from multiprocessing import Manager
import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt

def wc_xyz_min_max(wc_paths):
    start_time = time.time()
    with Manager() as manager:  ### 設定在 multiprocess 裡面 共用的 list
        ### 以下想 multiprocess
        ### global 的 list，應該就要用 share memory 了
        ch0_maxs = manager.list()  # []的概念
        ch0_mins = manager.list()  # []的概念
        ch1_maxs = manager.list()  # []的概念
        ch1_mins = manager.list()  # []的概念
        ch2_maxs = manager.list()  # []的概念
        ch2_mins = manager.list()  # []的概念
        core_amount = wc_amount // 150
        task_amount = wc_amount
        _wx_xyz_min_max_multiprocess(wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins, core_amount=core_amount, task_amount=task_amount)

        ### finish ~~ 開始show 結果囉
        ### 先整個list秀出來 大略看看 每個切出來的小task找的狀況
        print("ch0_maxs:", ch0_maxs)
        print("ch0_mins:", ch0_mins)
        print("ch1_maxs:", ch1_maxs)
        print("ch1_mins:", ch1_mins)
        print("ch2_maxs:", ch2_maxs)
        print("ch2_mins:", ch2_mins)
        print()

        ### list 轉 numpy，操作起來較方便(可以.max(), .min())
        ch0_maxs = np.array(ch0_maxs)
        ch0_mins = np.array(ch0_mins)
        ch1_maxs = np.array(ch1_maxs)
        ch1_mins = np.array(ch1_mins)
        ch2_maxs = np.array(ch2_maxs)
        ch2_mins = np.array(ch2_mins)

        ### 所有task 的min/max，即整個 DB 的min/max
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

    ### DewarpNet裡寫的數值： 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497 # RGB -> BGR
    """

def _wx_xyz_min_max_multiprocess(wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins, core_amount, task_amount):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_wc_xyz_min_max, task_args=[wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins], print_msg=True)

def _wc_xyz_min_max(start_index, amount, wc_paths, ch0_maxs, ch0_mins, ch1_maxs, ch1_mins, ch2_maxs, ch2_mins):
    '''
    主要做事的地方在這裡喔！
    '''
    wcs = []
    # start_index = 0
    # amount = 100
    for i, file_path in enumerate(tqdm(wc_paths[start_index:start_index + amount])):
        # print(i, file_path)
        wcs.append(cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED))  ### 這行就可以了！

    wcs  = np.array(wcs)  ### list 轉 numpy
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
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def wc_2D_3D_and_uv_visual(wc_paths, uv_paths, page_names_w_dir, dst_dir):
    start_time = time.time()
    wc_2d_dst_dir = dst_dir + "/" + "wc_visual_2D"
    wc_3d_dst_dir = dst_dir + "/" + "wc_visual_3D"
    uv_2d_dst_dir = dst_dir + "/" + "uv_visual_2D"
    Check_dir_exist_and_build(wc_2d_dst_dir)
    Check_dir_exist_and_build(wc_3d_dst_dir)
    Check_dir_exist_and_build(uv_2d_dst_dir)
    for dir_index in range(21):
        dir_name = dir_index + 1
        Check_dir_exist_and_build_new_dir(wc_2d_dst_dir + "/%i" % dir_name)
        Check_dir_exist_and_build_new_dir(wc_3d_dst_dir + "/%i" % dir_name)
        Check_dir_exist_and_build_new_dir(uv_2d_dst_dir + "/%i" % dir_name)


    core_amount = wc_amount // 70
    task_amount = wc_amount
    _wc_2D_3D_and_uv_visual_multiprocess(wc_paths, uv_paths, page_names_w_dir, wc_2d_dst_dir, wc_3d_dst_dir, uv_2d_dst_dir, core_amount=core_amount, task_amount=task_amount)


    for dir_index in range(21):
        dir_name = dir_index + 1
        Save_as_jpg(wc_2d_dst_dir + "/%i" % dir_name, wc_2d_dst_dir + "/%i" % dir_name, delete_ord_file=True)
        Save_as_jpg(wc_3d_dst_dir + "/%i" % dir_name, wc_3d_dst_dir + "/%i" % dir_name, delete_ord_file=True)
        Save_as_jpg(uv_2d_dst_dir + "/%i" % dir_name, uv_2d_dst_dir + "/%i" % dir_name, delete_ord_file=True)
    print("total_cost_time:", time.time() - start_time)

def _wc_2D_3D_and_uv_visual_multiprocess(wc_paths, uv_paths, page_names_w_dir, wc_2d_dst_dir, wc_3d_dst_dir, uv_2d_dst_dir, core_amount, task_amount):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_wc_2D_3D_and_uv_visual, task_args=[wc_paths, uv_paths, page_names_w_dir, wc_2d_dst_dir, wc_3d_dst_dir, uv_2d_dst_dir], print_msg=True)

def _wc_2D_3D_and_uv_visual(start_index, amount, wc_paths, uv_paths, page_names_w_dir, wc_2d_dst_dir, wc_3d_dst_dir, uv_2d_dst_dir):
    for i in tqdm(range(start_index, start_index + amount)):
        # print(i, file_path)
        uv = cv2.imread(uv_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)  ### 這行就可以了！
        mask = uv[..., 0]
        wc = cv2.imread(wc_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)  ### 這行就可以了！
        wc_3D_good_to_v = wc[..., ::-1]  ### 嘗試幾次後，這樣子比較好看

        wc_3d_dst_path = wc_3d_dst_dir + "/" + page_names_w_dir[i] + ".png"
        fig, ax = wc_3d_plot(wc_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), figsize=(5, 5))
        plt.savefig(wc_3d_dst_path)

        wc_2d_dst_path = wc_2d_dst_dir + "/" + page_names_w_dir[i] + ".png"
        fig, ax = wc_2d_plot(wc, figsize=(5, 5))
        plt.savefig(wc_2d_dst_path)
        plt.close()

        uv_2d_dst_path = uv_2d_dst_dir + "/" + page_names_w_dir[i] + ".png"
        fig, ax = uv_2d_plot(uv[..., ::-1], figsize=(5, 5))
        plt.savefig(uv_2d_dst_path)
        plt.close()

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def wc_uv_2_npy_knpy(doc3d, dst_dir, job_id=1, just_do_what_dir_num=None, core_amount=3):
    start_time = time.time()
    ### 定位出dir
    uv_dst_dir  = dst_dir + "/" + "1_uv"
    uv_npy_dir  = f"{uv_dst_dir}-1_npy"
    uv_vis_dir  = f"{uv_dst_dir}-2_visual"
    uv_knpy_dir = f"{uv_dst_dir}-3_knpy"

    wc_dst_dir     = dst_dir + "/" + "2_wc"
    wc_npy_dir     = f"{wc_dst_dir}-1_npy"
    wc_vis_2d_dir  = f"{wc_dst_dir}-2_2D_visual"
    wc_vis_3d_dir  = f"{wc_dst_dir}-3_3D_visual"
    W_w_M_npy_dir  = f"{wc_dst_dir}-4_W_w_M_npy"
    W_w_M_knpy_dir = f"{wc_dst_dir}-5_W_w_M_knpy"

    ### 因為檔案太多容易失敗， 所以就不要build_new_dir囉！
    if  (job_id == 1): Check_dir_exist_and_build(uv_npy_dir)
    elif(job_id == 2): Check_dir_exist_and_build(uv_vis_dir)
    elif(job_id == 3): Check_dir_exist_and_build(uv_knpy_dir)

    elif(job_id == 4): Check_dir_exist_and_build(wc_npy_dir)
    elif(job_id == 5): Check_dir_exist_and_build(wc_vis_2d_dir)
    elif(job_id == 6): Check_dir_exist_and_build(wc_vis_3d_dir)
    elif(job_id == 7): Check_dir_exist_and_build(W_w_M_npy_dir)
    elif(job_id == 8): Check_dir_exist_and_build(W_w_M_knpy_dir)

    dst_dict = {
        "uv_npy_dir"     : uv_npy_dir,
        "uv_vis_dir"     : uv_vis_dir,
        "uv_knpy_dir"    : uv_knpy_dir,
        "wc_npy_dir"     : wc_npy_dir,
        "wc_vis_2d_dir"  : wc_vis_2d_dir,
        "wc_vis_3d_dir"  : wc_vis_3d_dir,
        "W_w_M_npy_dir"  : W_w_M_npy_dir,
        "W_w_M_knpy_dir" : W_w_M_knpy_dir,
    }

    task_amount      = len(doc3d.wc_paths)
    task_start_index = 0
    if(just_do_what_dir_num is not None):
        just_do_what_dir_index = just_do_what_dir_num - 1

        task_amount      = doc3d.dir_data_amounts     [just_do_what_dir_index]
        task_start_index = doc3d.dir_data_amounts_acc [just_do_what_dir_index - 1]
        if(just_do_what_dir_num == 1): task_start_index = 0


    _wc_uv_2_npy_knpy_multiprocess(doc3d, dst_dict, core_amount=core_amount, job_id=job_id, task_amount=task_amount, task_start_index=task_start_index)

    if  (job_id == 1): Save_as_jpg(uv_npy_dir     , uv_npy_dir     , delete_ord_file=True)
    elif(job_id == 2): Save_as_jpg(uv_vis_dir     , uv_vis_dir     , delete_ord_file=True)
    elif(job_id == 3): Save_as_jpg(uv_knpy_dir    , uv_knpy_dir    , delete_ord_file=True)
    elif(job_id == 4): Save_as_jpg(wc_npy_dir     , wc_npy_dir     , delete_ord_file=True)
    elif(job_id == 5): Save_as_jpg(wc_vis_2d_dir  , wc_vis_2d_dir  , delete_ord_file=True)
    elif(job_id == 6): Save_as_jpg(wc_vis_3d_dir  , wc_vis_3d_dir  , delete_ord_file=True)
    elif(job_id == 7): Save_as_jpg(W_w_M_npy_dir  , W_w_M_npy_dir  , delete_ord_file=True)
    elif(job_id == 8): Save_as_jpg(W_w_M_knpy_dir , W_w_M_knpy_dir , delete_ord_file=True)
    print("total_cost_time:", time.time() - start_time)

def _wc_uv_2_npy_knpy_multiprocess(doc3d, dst_dict, job_id, core_amount, task_amount, task_start_index):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_wc_uv_2_npy_knpy, task_start_index=task_start_index, task_args=[doc3d, dst_dict, job_id], print_msg=True)

def _wc_uv_2_npy_knpy(start_index, amount, doc3d, dst_dict, job_id):
    for i in tqdm(range(start_index, start_index + amount)):
        # print(i, file_path)
        ####### 定位出path
        uv_npy_path  = dst_dict["uv_npy_dir"]        + "/" + doc3d.page_names_w_dir_combine[i] + ".npy"
        uv_vis_path  = dst_dict["uv_vis_dir"]        + "/" + doc3d.page_names_w_dir_combine[i] + ".png"
        uv_knpy_path = dst_dict["uv_knpy_dir"]       + "/" + doc3d.page_names_w_dir_combine[i] + ".knpy"

        wc_npy_path     = dst_dict["wc_npy_dir"]     + "/" + doc3d.page_names_w_dir_combine[i] + ".npy"
        wc_vis_2d_path  = dst_dict["wc_vis_2d_dir"]  + "/" + doc3d.page_names_w_dir_combine[i] + ".png"
        wc_vis_3d_path  = dst_dict["wc_vis_3d_dir"]  + "/" + doc3d.page_names_w_dir_combine[i] + ".png"
        W_w_M_npy_path  = dst_dict["W_w_M_npy_dir"]  + "/" + doc3d.page_names_w_dir_combine[i] + ".npy"
        W_w_M_knpy_path = dst_dict["W_w_M_knpy_dir"] + "/" + doc3d.page_names_w_dir_combine[i] + ".knpy"

        ####### 做事情的地方
        uv              = get_exr(doc3d.uv_paths[i])
        mask            = uv[..., 0]
        wc              = get_exr(doc3d.wc_paths[i])
        wc_3D_good_to_v = wc             [..., 0:3]   ### 嘗試幾次後，這樣子比較好看
        wc_3D_good_to_v = wc_3D_good_to_v[..., ::-1]  ### 嘗試幾次後，這樣子比較好看
        W_w_M           = wc.copy()
        W_w_M[..., 3]   = mask

        uv              = uv   .astype(np.float32)
        wc              = wc   .astype(np.float32)
        W_w_M           = W_w_M.astype(np.float32)

        ####### 存到相對應的位置
        ##### uv part
        if(job_id == 1):
            ### uv-1_npy
            np.save(uv_npy_path, uv)

        elif(job_id == 2):
            ### uv-2_visual
            fig, ax = uv_2d_plot(uv[..., ::-1], figsize=(5, 5))
            plt.savefig(uv_vis_path)
            plt.close()

        elif(job_id == 3):
            ### uv-3_knpy
            Save_npy_path_as_knpy(src_path=uv_npy_path, dst_path=uv_knpy_path)

        elif(job_id == 4):
            ##### wc part
            ### wc-1_npy
            np.save(wc_npy_path, wc)

        elif(job_id == 5):
            ### wc-2_2D_visual
            fig, ax = wc_2d_plot(wc, figsize=(5, 5))
            plt.savefig(wc_vis_2d_path)
            plt.close()

        elif(job_id == 6):
            ### wc-3_3D_visual
            fig, ax = wc_3d_plot(wc_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), figsize=(5, 5))
            plt.savefig(wc_vis_3d_path)
            plt.close()

        elif(job_id == 7):
            ### wc-4_W_w_M_npy
            np.save(W_w_M_npy_path, W_w_M)

        elif(job_id == 8):
            ### wc-5_W_w_M_knpy
            Save_npy_path_as_knpy(src_path=W_w_M_npy_path, dst_path=W_w_M_knpy_path)
        

if(__name__ == "__main__"):
    from step0_Doc3D_obj import  using_doc3D
    '''1_244_1-cp_Page_0995-mpT0001.exr'''
    ### 取得 wc_paths
    wc_paths = using_doc3D.wc_paths
    uv_paths = using_doc3D.uv_paths
    page_names_w_dir = using_doc3D.page_names_w_dir
    # for wc_path in wc_paths: print(wc_path)

    ### 設定 要處理的數量
    # wc_amount = 2000 #len(wc_paths)  ### 少量測試時用的
    wc_amount = len(wc_paths)
    # print("wc_amount:", wc_amount)
    ############################################################################################################

    ### 分析1：找 整個DB 所有 wc 各個channel 的 min/max
    # wc_xyz_min_max(wc_paths)

    ### 分析2
    # wc_2D_3D_and_uv_visual(wc_paths, uv_paths, page_names_w_dir, dst_dir=r"H:")

    ### 做事1
    ### 101838
    ### 102064
    SSD_dst_dir = "F:/kong_doc3d"
    HDD_dst_dir = "K:\kong_doc3d"
    Check_dir_exist_and_build(HDD_dst_dir)  ### 一定要先建立 HDD_dst_dir， 要不然LOG檔沒地方存會報錯
    for dir_num in range(21):
        if( dir_num < 15 ): continue
        just_do_what_dir_num = dir_num + 1

        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=1, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=3, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=4, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=7, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=8, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=2, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=5, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir="F:/kong_doc3d", job_id=6, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)

        os.system(f"robocopy {SSD_dst_dir}/1_uv-1_npy             {HDD_dst_dir}/1_uv-1_npy             /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-1_uv-1_npy.txt"             % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/1_uv-2_visual          {HDD_dst_dir}/1_uv-2_visual          /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-1_uv-2_visual.txt"          % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/1_uv-3_knpy            {HDD_dst_dir}/1_uv-3_knpy            /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-1_uv-3_knpy.txt"            % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-1_npy             {HDD_dst_dir}/2_wc-1_npy             /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-2_wc-1_npy.txt"             % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-2_2D_visual       {HDD_dst_dir}/2_wc-2_2D_visual       /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-2_wc-2_2D_visual.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-3_3D_visual       {HDD_dst_dir}/2_wc-3_3D_visual       /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-2_wc-3_3D_visual.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-4_W_w_M_npy       {HDD_dst_dir}/2_wc-4_W_w_M_npy       /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-2_wc-4_W_w_M_npy.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-5_W_w_M_knpy      {HDD_dst_dir}/2_wc-5_W_w_M_knpy      /MOVE /E /MT:100 /LOG:{HDD_dst_dir}/robocopy-%02i-2_wc-5_W_w_M_knpy.txt"      % (just_do_what_dir_num))

        '''
        robocopy F:/kong_doc3d/1_uv-1_npy             K:\kong_doc3d/1_uv-1_npy            /MOVE /E /MT:100 /LOG:K:\kong_doc3d/robocopy-2-1_uv-1_npy.txt"

        command = f"robocopy {SSD_dst_dir} {HDD_dst_dir} /MOVE /E /MT:100 /LOG:{HDD_dst_dir}\robocopy_{just_do_what_num}.txt"
        ### 可參考：https://eric0806.blogspot.com/2013/02/robocopy.html
            /MOVE：移動檔案和目錄 (複製後從來源刪除)。
            /E   ：複製子目錄，包括空的子目錄。
            /M   ：以 n 個執行緒執行多執行緒複製 (預設值為 8)。
            /LOG ：輸出狀態至記錄檔 (覆寫現有的記錄檔)。
        os.system(command)
        '''
