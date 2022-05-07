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
from kong_util.build_dataset_combine import Check_dir_exist_and_build
from kong_util.flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img

import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import shutil

def build_sep_dir(dir_path):
    for i in range(21):
        Check_dir_exist_and_build(f"{dir_path}/%02i" % (i + 1))
############################################################################################################################################
def dis_img_and_rec_hope(doc3d, dst_dir, use_sep_name=False, job_id=1, just_do_what_dir_num=None, core_amount=3):
    start_time = time.time()
    dis_img_dir  = dst_dir + "/" + "0_dis_img"
    rec_hope_dir = dst_dir + "/" + "0_rec_hope"

    ### 因為檔案太多容易失敗， 所以就不要build_new_dir囉！
    if  (job_id == 1): Check_dir_exist_and_build(dis_img_dir)
    elif(job_id == 2): Check_dir_exist_and_build(rec_hope_dir)

    if(job_id == 1 and use_sep_name is True): build_sep_dir(dis_img_dir)
    if(job_id == 2 and use_sep_name is True): build_sep_dir(rec_hope_dir)

    dst_dict = {
        "dis_img_dir"  : dis_img_dir,
        "rec_hope_dir" : rec_hope_dir,
    }

    task_amount      = len(doc3d.img_paths)
    task_start_index = 0
    if(just_do_what_dir_num is not None):  ### 看有沒有指定要做 1 ~ 21 中的哪個資料夾
        just_do_what_dir_index = just_do_what_dir_num - 1

        task_amount      = doc3d.dir_data_amounts     [just_do_what_dir_index]      ### 你指定的 1~21 指定的資料夾 的該個資料夾 有幾筆資料要處理
        task_start_index = doc3d.dir_data_amounts_acc [just_do_what_dir_index - 1]  ### 你指定的 1~21 指定的資料夾 的該個資料夾 在整體list中 的 起始位置
        if(just_do_what_dir_num == 1): task_start_index = 0                         ### 如果你指定第 1 個資料夾， 會變成 doc3d.dir_data_amounts_acc [-1] 就取錯了喔！ 起始位置為 0才對， 手動設定0這樣子


    _dis_img_and_rec_hope_multiprocess(doc3d, dst_dict, use_sep_name=use_sep_name, job_id=job_id, core_amount=core_amount, task_amount=task_amount, task_start_index=task_start_index)

    print("total_cost_time:", time.time() - start_time)

def _dis_img_and_rec_hope_multiprocess(doc3d, dst_dict, use_sep_name, job_id, core_amount, task_amount, task_start_index):
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_dis_img_and_rec_hope, task_start_index=task_start_index, task_args=[doc3d, dst_dict, use_sep_name, job_id], print_msg=False)

def _dis_img_and_rec_hope(start_index, amount, doc3d, dst_dict, use_sep_name, job_id):
    for i in tqdm(range(start_index, start_index + amount)):
        # print(i, file_path)
        ####### 定位出path
        if(use_sep_name is False): use_what_name = doc3d.page_names_w_dir_combine[i]
        else                     : use_what_name = doc3d.page_names_w_dir_combine_sep[i]

        dis_img_path  = dst_dict["dis_img_dir"]  + "/" + use_what_name + ".png"
        rec_hope_path = dst_dict["rec_hope_dir"] + "/" + use_what_name + ".png"


        ####### 做事情的地方
        if  (job_id == 1):
            shutil.copy(doc3d.img_paths[i], dis_img_path)

        elif(job_id == 2):
            dis_img         = cv2.imread(doc3d.img_paths[i])
            uv              = get_exr(doc3d.uv_paths [i])
            bm  = use_flow_to_get_bm(flow=uv, flow_scale=448)
            rec = use_bm_to_rec_img(bm=bm, dis_img=dis_img, flow_scale=448)
            cv2.imwrite(rec_hope_path, rec)

if(__name__ == "__main__"):
    from step0_Doc3D_obj import  using_doc3D
    ###########################################################################################################
    ### 做事1
    ### 101838
    ### 102064 - 2 ( 去掉 21/2_431_3-cp_Page_0802-Pum0001 和 21/556_7-ny_Page_183-cvM0001)
    SSD_dst_dir = "F:/kong_doc3d/train"
    HDD_dst_dir = "I:/kong_doc3d/train"
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src_log_dir = f"{SSD_dst_dir}/LOG-{current_time}"
    dst_log_dir = f"{HDD_dst_dir}/LOG-{current_time}"
    use_sep_name = True

    Check_dir_exist_and_build(src_log_dir)  ### 一定要先建立 src_log_dir 要不然LOG檔沒地方存會報錯
    for dir_num in range(21):
        just_do_what_dir_num = dir_num + 1

        dis_img_and_rec_hope(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=1, just_do_what_dir_num=just_do_what_dir_num, core_amount=2)
        dis_img_and_rec_hope(using_doc3D, dst_dir=SSD_dst_dir, use_sep_name=use_sep_name, job_id=2, just_do_what_dir_num=just_do_what_dir_num, core_amount=10)

        os.system(f"robocopy {SSD_dst_dir}/0_dis_img  {HDD_dst_dir}/0_dis_img   /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-0_dis_img.txt"   % (just_do_what_dir_num))
        os.system(f"robocopy {SSD_dst_dir}/0_rec_hope {HDD_dst_dir}/0_rec_hope  /MOVE /E /MT:100 /LOG:{src_log_dir}/robocopy-%02i-0_rec_hope.txt"  % (just_do_what_dir_num))
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
