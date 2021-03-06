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

import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt

from step0_Doc3D_obj  import  real_doc3D
from step0_Kong_Doc3D import  kong_doc3D

def build_sep_dir(dir_path):
    for i in range(21):
        Check_dir_exist_and_build(f"{dir_path}/%02i" % (i + 1))
############################################################################################################################################
def wc_uv_2_npy_knpy(doc3d, dst_dir, use_sep_name=False, job_id=1, ord_z_bot=None, ord_z_top=None, just_do_what_dir_num=None, core_amount=3):
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
    if("1_uv-1_npy"        in job_id): Check_dir_exist_and_build(uv_npy_dir)
    if("1_uv-2_visual"     in job_id): Check_dir_exist_and_build(uv_vis_dir)
    if("1_uv-3_knpy"       in job_id): Check_dir_exist_and_build(uv_knpy_dir)
    if("2_wc-1_npy"        in job_id): Check_dir_exist_and_build(wc_npy_dir)
    if("2_wc-2_2D_visual"  in job_id): Check_dir_exist_and_build(wc_vis_2d_dir)
    if("2_wc-3_3D_visual"  in job_id): Check_dir_exist_and_build(wc_vis_3d_dir)
    if("2_wc-4_W_w_M_npy"  in job_id): Check_dir_exist_and_build(W_w_M_npy_dir)
    if("2_wc-5_W_w_M_knpy" in job_id): Check_dir_exist_and_build(W_w_M_knpy_dir)

    if("1_uv-1_npy"        in job_id and use_sep_name is True): build_sep_dir(uv_npy_dir)
    if("1_uv-2_visual"     in job_id and use_sep_name is True): build_sep_dir(uv_vis_dir)
    if("1_uv-3_knpy"       in job_id and use_sep_name is True): build_sep_dir(uv_knpy_dir)

    if("2_wc-1_npy"        in job_id and use_sep_name is True): build_sep_dir(wc_npy_dir)
    if("2_wc-2_2D_visual"  in job_id and use_sep_name is True): build_sep_dir(wc_vis_2d_dir)
    if("2_wc-3_3D_visual"  in job_id and use_sep_name is True): build_sep_dir(wc_vis_3d_dir)
    if("2_wc-4_W_w_M_npy"  in job_id and use_sep_name is True): build_sep_dir(W_w_M_npy_dir)
    if("2_wc-5_W_w_M_knpy" in job_id and use_sep_name is True): build_sep_dir(W_w_M_knpy_dir)

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

    if  (type(doc3d) == type(real_doc3D)): task_amount = len(doc3d.wc_paths)
    elif(type(doc3d) == type(kong_doc3D)): task_amount = len(doc3d.wc_npy_paths)
    task_start_index = 0
    if(just_do_what_dir_num is not None):
        just_do_what_dir_index = just_do_what_dir_num - 1

        task_amount      = doc3d.dir_data_amounts     [just_do_what_dir_index]
        task_start_index = doc3d.dir_data_amounts_acc [just_do_what_dir_index - 1]
        if(just_do_what_dir_num == 1): task_start_index = 0


    _wc_uv_2_npy_knpy_multiprocess(doc3d, dst_dict, use_sep_name=use_sep_name, job_id=job_id, ord_z_bot=ord_z_bot, ord_z_top=ord_z_top, core_amount=core_amount, task_amount=task_amount, task_start_index=task_start_index)

    if("1_uv-1_npy"        in job_id): Save_as_jpg(uv_npy_dir     , uv_npy_dir     , delete_ord_file=True)
    if("1_uv-2_visual"     in job_id): Save_as_jpg(uv_vis_dir     , uv_vis_dir     , delete_ord_file=True)
    if("1_uv-3_knpy"       in job_id): Save_as_jpg(uv_knpy_dir    , uv_knpy_dir    , delete_ord_file=True)
    if("2_wc-1_npy"        in job_id): Save_as_jpg(wc_npy_dir     , wc_npy_dir     , delete_ord_file=True)
    if("2_wc-2_2D_visual"  in job_id): Save_as_jpg(wc_vis_2d_dir  , wc_vis_2d_dir  , delete_ord_file=True)
    if("2_wc-3_3D_visual"  in job_id): Save_as_jpg(wc_vis_3d_dir  , wc_vis_3d_dir  , delete_ord_file=True)
    if("2_wc-4_W_w_M_npy"  in job_id): Save_as_jpg(W_w_M_npy_dir  , W_w_M_npy_dir  , delete_ord_file=True)
    if("2_wc-5_W_w_M_knpy" in job_id): Save_as_jpg(W_w_M_knpy_dir , W_w_M_knpy_dir , delete_ord_file=True)
    print("total_cost_time:", time.time() - start_time)

def _wc_uv_2_npy_knpy_multiprocess(doc3d, dst_dict, use_sep_name, job_id, ord_z_bot, ord_z_top, core_amount, task_amount, task_start_index):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_wc_uv_2_npy_knpy, task_start_index=task_start_index, task_args=[doc3d, dst_dict, use_sep_name, job_id, ord_z_bot, ord_z_top], print_msg=False)

def _wc_uv_2_npy_knpy(start_index, amount, doc3d, dst_dict, use_sep_name, job_id, ord_z_bot, ord_z_top):
    for i in tqdm(range(start_index, start_index + amount)):
        # print(i, file_path)
        if(use_sep_name is False): use_what_name = doc3d.page_names_w_dir_combine[i]
        else                     : use_what_name = doc3d.page_names_w_dir_combine_sep[i]
        ####### 定位出path
        uv_npy_path  = dst_dict["uv_npy_dir"]        + "/" + use_what_name + ".npy"
        uv_vis_path  = dst_dict["uv_vis_dir"]        + "/" + use_what_name + ".png"
        uv_knpy_path = dst_dict["uv_knpy_dir"]       + "/" + use_what_name + ".knpy"

        wc_npy_path     = dst_dict["wc_npy_dir"]     + "/" + use_what_name + ".npy"
        wc_vis_2d_path  = dst_dict["wc_vis_2d_dir"]  + "/" + use_what_name + ".png"
        wc_vis_3d_path  = dst_dict["wc_vis_3d_dir"]  + "/" + use_what_name + ".png"
        W_w_M_npy_path  = dst_dict["W_w_M_npy_dir"]  + "/" + use_what_name + ".npy"
        W_w_M_knpy_path = dst_dict["W_w_M_knpy_dir"] + "/" + use_what_name + ".knpy"

        ####### 做事情的地方
        if  (type(doc3d) == type(real_doc3D)): uv = get_exr(doc3d.uv_paths[i])
        elif(type(doc3d) == type(kong_doc3D)): uv = np.load(doc3d.uv_npy_paths[i])
        mask            = uv[..., 0]

        ### 原始 Doc3D 需要 1.加入M, 2.x, y 對調
        if  (type(doc3d) == type(real_doc3D)):
            ### 1. 加入M
            wc_zxy = get_exr(doc3d.wc_paths[i])  ### z, x, y, ?
            wc_zxy             = get_exr(doc3d.wc_paths[i])  ### z, x, y, ?
            Wzxy_w_M           = wc_zxy.copy()
            Wzxy_w_M[..., 3]   = mask

            ### 2. x, y 對調
            Wx = Wzxy_w_M[..., 1:2].copy()
            Wy = Wzxy_w_M[..., 2:3].copy()
            Wzyx_w_M = Wzxy_w_M.copy()
            Wzyx_w_M[..., 1:2] = Wy.copy()
            Wzyx_w_M[..., 2:3] = Wx.copy()
            Wzyx = Wzyx_w_M[..., :3].copy()

        ### Kong_Doc3D 第一版 已經 做完 加入M 和 xy對調了， 所以直接讀出來即可
        elif(type(doc3d) == type(kong_doc3D)):
            Wzyx     = np.load(doc3d.wc_npy_paths[i])
            Wzyx_w_M = np.load(doc3d.W_w_M_npy_paths[i])

        ##### 原始Doc3D 和 Kong_Doc3D 第一版 都要做， 在Kong_Doc3D第一版實際訓練下去才發現這個問題！ 1.x軸相反, 2.z置中
        ### 1. x值reverse
        # fig, ax = wc_3d_plot(Wzyx[..., ::-1], mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
        Wzyx    [..., 2] *= -1
        # fig, ax = wc_3d_plot(Wzyx[..., ::-1], mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z

        # Wzyx_3D_good_to_v = Wzyx_w_M[...,0 : 3].copy()   ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
        # Wzyx_3D_good_to_v = Wzyx_3D_good_to_v[...,::-1]  ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
        # fig, ax = wc_3d_plot(Wzyx_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
        Wzyx_w_M[..., 2] *= -1
        # Wzyx_3D_good_to_v = Wzyx_w_M[...,0 : 3].copy()   ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
        # Wzyx_3D_good_to_v = Wzyx_3D_good_to_v[...,::-1]  ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
        # fig, ax = wc_3d_plot(Wzyx_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
        # plt.show()

        ### 2. z 置中 (需要先知道 原始 doc3d 的 z_min/max 喔！)
        if(None not in [ord_z_bot, ord_z_top]):
            # ord_z_range = ord_z_top - ord_z_bot
            Z = Wzyx_w_M[..., 0:1].copy()
            Z_w_M = Z[mask.astype(np.bool)]
            Z_w_M_top = Z_w_M.max()
            Z_w_M_bot = Z_w_M.min()
            Z_w_M_top_res = abs( ord_z_top - Z_w_M_top )
            Z_w_M_bot_res = abs( ord_z_bot - Z_w_M_bot )
            z_move = (Z_w_M_bot_res - Z_w_M_top_res) / 2
            # fig, ax = wc_3d_plot(Wzyx[..., ::-1], mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
            Wzyx    [..., 0] -= (z_move * mask)
            # fig, ax = wc_3d_plot(Wzyx[..., ::-1], mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z

            # Wzyx_3D_good_to_v = Wzyx_w_M[...,0 : 3].copy()   ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
            # Wzyx_3D_good_to_v = Wzyx_3D_good_to_v[...,::-1]  ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
            # fig, ax = wc_3d_plot(Wzyx_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
            Wzyx_w_M[..., 0] -= (z_move * mask )
            # Wzyx_3D_good_to_v = Wzyx_w_M[...,0 : 3].copy()   ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
            # Wzyx_3D_good_to_v = Wzyx_3D_good_to_v[...,::-1]  ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化
            # fig, ax = wc_3d_plot(Wzyx_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
            # plt.show()


        Wzyx_3D_good_to_v = Wzyx[..., ::-1]  ### 嘗試幾次後，這樣子比較好看，剛好變 xyz很好視覺化

        ### 用這個真的確認： doc3D 他們的 W 是 z, x, y ！ 我的是 z, y, x
        # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(5 * 2, 5))
        # ax[0, 0].imshow(Wzxy_w_M[..., 0:1])
        # ax[0, 1].imshow(Wzxy_w_M[..., 1:2])
        # ax[0, 2].imshow(Wzxy_w_M[..., 2:3])
        # ax[1, 0].imshow(Wzyx_w_M[..., 0:1])
        # ax[1, 1].imshow(Wzyx_w_M[..., 1:2])
        # ax[1, 2].imshow(Wzyx_w_M[..., 2:3])
        # plt.show()

        ### 轉 dtype
        uv       = uv       .astype(np.float32)
        Wzyx     = Wzyx     .astype(np.float32)
        Wzyx_w_M = Wzyx_w_M .astype(np.float32)

        ####### 存到相對應的位置
        ##### uv part
        if("1_uv-1_npy" in job_id):
            ### uv-1_npy
            np.save(uv_npy_path, uv)

        if("1_uv-2_visual" in job_id):
            ### uv-2_visual
            fig, ax = uv_2d_plot(uv[..., ::-1], figsize=(5, 5))
            plt.savefig(uv_vis_path)
            plt.close()

        if("1_uv-3_knpy" in job_id):
            ### uv-3_knpy
            Save_npy_path_as_knpy(src_path=uv_npy_path, dst_path=uv_knpy_path)

        if("2_wc-1_npy" in job_id):
            ##### wc part
            ### wc-1_npy
            np.save(wc_npy_path, Wzyx)

        if("2_wc-2_2D_visual" in job_id):
            ### wc-2_2D_visual
            fig, ax = wc_2d_plot(Wzyx, figsize=(5, 5))
            plt.savefig(wc_vis_2d_path)
            plt.close()

        if("2_wc-3_3D_visual" in job_id):
            ### wc-3_3D_visual
            fig, ax = wc_3d_plot(Wzyx_3D_good_to_v, mask, fewer_point=True, small_size=(300, 300), ax_size=5, ch0_min=-1.2280148, ch0_max=1.2387834, ch1_min=-1.2410645, ch1_max=1.2485291, ch2_min=-0.67187124, ch2_max=0.63452387)  ### ch0:x, ch1:y, ch2:z
            plt.savefig(wc_vis_3d_path)
            # plt.show()
            plt.close()

        if("2_wc-4_W_w_M_npy" in job_id):
            ### wc-4_W_w_M_npy
            np.save(W_w_M_npy_path, Wzyx_w_M)

        if("2_wc-5_W_w_M_knpy" in job_id):
            ### wc-5_W_w_M_knpy
            Save_npy_path_as_knpy(src_path=W_w_M_npy_path, dst_path=W_w_M_knpy_path)

if(__name__ == "__main__"):
    using_doc3D = kong_doc3D
    ### 做事1
    ### 101838
    ### 102064 - 2 ( 去掉 21/2_431_3-cp_Page_0802-Pum0001 和 21/556_7-ny_Page_183-cvM0001)
    SSD_dst_dir = "F:/kong_doc3d/train"
    HDD_dst_dir = "E:/kong_doc3d/train"
    use_sep_name = True
    just_do_what_dir_nums = [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                              11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21 ]
    ###########################################################################################################
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src_log_dir = f"{SSD_dst_dir}/LOG-{current_time}"
    dst_log_dir = f"{HDD_dst_dir}/LOG-{current_time}"
    Check_dir_exist_and_build(src_log_dir)  ### 一定要先建立 HDD_dst_dir， 要不然LOG檔沒地方存會報錯

    for just_do_what_dir_num in just_do_what_dir_nums:
        # if( just_do_what_dir_num != 19): continue

        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["1_uv-1_npy"        ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["1_uv-3_knpy"       ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-4_W_w_M_npy"  ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-5_W_w_M_knpy" ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-1_npy"        ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["1_uv-2_visual"     ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-2_2D_visual"  ], just_do_what_dir_num=just_do_what_dir_num, core_amount=25)
        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-3_3D_visual"  ], just_do_what_dir_num=just_do_what_dir_num, core_amount=100)

        wc_uv_2_npy_knpy(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=["2_wc-4_W_w_M_npy", "2_wc-5_W_w_M_knpy"], ord_z_bot=-0.67187124, ord_z_top=0.63452387, just_do_what_dir_num=just_do_what_dir_num, core_amount=25)


        os.system(f"robocopy {SSD_dst_dir}/1_uv-1_npy             {HDD_dst_dir}/1_uv-1_npy             /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-1_uv-1_npy.txt"             % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/1_uv-2_visual          {HDD_dst_dir}/1_uv-2_visual          /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-1_uv-2_visual.txt"          % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/1_uv-3_knpy            {HDD_dst_dir}/1_uv-3_knpy            /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-1_uv-3_knpy.txt"            % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-1_npy             {HDD_dst_dir}/2_wc-1_npy             /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-2_wc-1_npy.txt"             % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-2_2D_visual       {HDD_dst_dir}/2_wc-2_2D_visual       /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-2_wc-2_2D_visual.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-3_3D_visual       {HDD_dst_dir}/2_wc-3_3D_visual       /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-2_wc-3_3D_visual.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-4_W_w_M_npy       {HDD_dst_dir}/2_wc-4_W_w_M_npy       /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-2_wc-4_W_w_M_npy.txt"       % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/2_wc-5_W_w_M_knpy      {HDD_dst_dir}/2_wc-5_W_w_M_knpy      /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-2_wc-5_W_w_M_knpy.txt"      % (just_do_what_dir_num))
        os.system(f"robocopy {src_log_dir} {dst_log_dir} /E")

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
