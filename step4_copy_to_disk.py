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
from kong_util.build_dataset_combine import Check_dir_exist_and_build
import os
import datetime
if(__name__ == "__main__"):
    ###########################################################################################################
    ### 做事1
    ### 101838
    ### 102064
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src_dir     = "E:/kong_doc3d/train"
    dst_dir     = "J:/kong_doc3d/train"
    src_log_dir = f"{src_dir}/LOG-{current_time}"
    dst_log_dir = f"{dst_dir}/LOG-{current_time}"
    Check_dir_exist_and_build(src_log_dir)  ### 一定要先建立 HDD_dst_dir， 要不然LOG檔沒地方存會報錯

    ### use_sep_name = False 的 copy方式
    # os.system(f"robocopy {src_dir}/0_dis_img          {dst_dir}/0_dis_img          /E /MT:100 /UNILOG:{src_log_dir}/robocopy-0_dis_img.txt"        )
    # os.system(f"robocopy {src_dir}/0_rec_hope         {dst_dir}/0_rec_hope         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-0_rec_hope.txt"       )
    # os.system(f"robocopy {src_dir}/1_uv-3_knpy        {dst_dir}/1_uv-3_knpy        /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-3_knpy.txt"      )
    # os.system(f"robocopy {src_dir}/2_wc-5_W_w_M_knpy  {dst_dir}/2_wc-5_W_w_M_knpy  /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-5_W_w_M_knpy.txt")
    # os.system(f"robocopy {src_dir}/2_wc-1_npy         {dst_dir}/2_wc-1_npy         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-1_npy.txt"       )
    # os.system(f"robocopy {src_dir}/1_uv-1_npy         {dst_dir}/1_uv-1_npy         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-1_npy.txt"       )
    # os.system(f"robocopy {src_dir}/2_wc-4_W_w_M_npy   {dst_dir}/2_wc-4_W_w_M_npy   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-4_W_w_M_npy.txt" )
    # os.system(f"robocopy {src_dir}/1_uv-2_visual      {dst_dir}/1_uv-2_visual      /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-2_visual.txt"    )
    # os.system(f"robocopy {src_dir}/2_wc-2_2D_visual   {dst_dir}/2_wc-2_2D_visual   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-2_2D_visual.txt" )
    # os.system(f"robocopy {src_dir}/2_wc-3_3D_visual   {dst_dir}/2_wc-3_3D_visual   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-3_3D_visual.txt" )
    # os.system(f"robocopy {src_log_dir} {dst_log_dir} /E")


    ### use_sep_name = True 的 copy方式 
    want_what_dirs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                      '11', '12', '13', '14', '15',
                      '16', '17', '18', '19', '20', '21',
                     ]
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/0_dis_img/{what_dir}          {dst_dir}/0_dis_img/{what_dir}          /E /MT:100 /UNILOG:{src_log_dir}/robocopy-0_dis_img-{what_dir}.txt"        )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/1_uv-3_knpy/{what_dir}        {dst_dir}/1_uv-3_knpy/{what_dir}        /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-3_knpy-{what_dir}.txt"      )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/2_wc-5_W_w_M_knpy/{what_dir}  {dst_dir}/2_wc-5_W_w_M_knpy/{what_dir}  /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-5_W_w_M_knpy-{what_dir}.txt")
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/0_rec_hope/{what_dir}         {dst_dir}/0_rec_hope/{what_dir}         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-0_rec_hope-{what_dir}.txt"       )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/2_wc-1_npy/{what_dir}         {dst_dir}/2_wc-1_npy/{what_dir}         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-1_npy-{what_dir}.txt"       )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/1_uv-1_npy/{what_dir}         {dst_dir}/1_uv-1_npy/{what_dir}         /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-1_npy-{what_dir}.txt"       )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/2_wc-4_W_w_M_npy/{what_dir}   {dst_dir}/2_wc-4_W_w_M_npy/{what_dir}   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-4_W_w_M_npy-{what_dir}.txt" )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/1_uv-2_visual/{what_dir}      {dst_dir}/1_uv-2_visual/{what_dir}      /E /MT:100 /UNILOG:{src_log_dir}/robocopy-1_uv-2_visual-{what_dir}.txt"    )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/2_wc-2_2D_visual/{what_dir}   {dst_dir}/2_wc-2_2D_visual/{what_dir}   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-2_2D_visual-{what_dir}.txt" )
    for what_dir in want_what_dirs: os.system(f"robocopy {src_dir}/2_wc-3_3D_visual/{what_dir}   {dst_dir}/2_wc-3_3D_visual/{what_dir}   /E /MT:100 /UNILOG:{src_log_dir}/robocopy-2_wc-3_3D_visual-{what_dir}.txt" )
    os.system(f"robocopy {src_log_dir} {dst_log_dir} /E")
