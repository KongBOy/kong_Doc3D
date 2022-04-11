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
import os
import numpy as np
import shutil
from tqdm import tqdm

from kong_util.build_dataset_combine import Check_dir_exist_and_build_new_dir, Check_dir_exist_and_build

class Doc3D:
    '''
    page_names: 這是給 DewarpNet 用的喔！我自己的話應該直接用 page_paths， 哈哈page_names雖然看起來很多此一舉但不能刪喔！！！
    wc_paths: wc 各別的路徑～
    uv_paths: uv 各別的路徑～
    '''

    def __init__(self, root):
        self.db_root = root  ### Doc3D
        self._page_names = None
        self._img_paths  = None
        self._wc_paths   = None
        self._uv_paths   = None
        # self.get_page_names_and_paths()   ### 不要在這邊直接呼叫，我下面會建立 real_doc3D物件， 在這邊呼叫就代表一定要插 2T Doc3D 硬碟 才能跑！
        ### 所以改成 跟try_forward 一樣， 寫成 property 的感覺囉！

    @property
    def page_names(self):
        if(self._page_names is None): self.get_page_names_and_paths()
        return self._page_names

    @property
    def img_paths(self):
        if(self._img_paths is None): self.get_page_names_and_paths()
        return self._img_paths

    @property
    def wc_paths(self):
        if(self._wc_paths is None): self.get_page_names_and_paths()
        return self._wc_paths

    @property
    def uv_paths(self):
        if(self._uv_paths is None): self.get_page_names_and_paths()
        return self._uv_paths

    def get_page_names_and_paths(self):
        print("get_page_names_and_paths here~~~~~ should just be used only once!應該只會被用到一次")
        self._page_names = []
        self._img_paths  = []
        self._wc_paths   = []
        self._uv_paths   = []
        train_name_dirs = [self.db_root + "/img/%i" % i for i in range(1, 22)]  ### ["F:/swat3D/1", "F:/swat3D/2", ..., "F:/swat3D/21"]
        for i , name_dir in enumerate(train_name_dirs):
            dir_id = i + 1
            names = os.listdir(name_dir)  ### [1000_1-cp_Page_0179-r8T0001.png, 105_3-ny_Page_850-02a0001.png, ... ]
            for name in names:
                page_name = str(dir_id) + "/" + name[: -4]  ### 去掉副檔名 和 加上 dir_id，比如 21/1000_1-cp_Page_0179-r8T0001 就是一個 page_name
                self._page_names.append(page_name)  ### [21/1000_1-cp_Page_0179-r8T0001, 21/105_3-ny_Page_850-02a0001, ... ]
                self._img_paths .append(self.db_root + "/img/" + page_name + ".png" )
                self._wc_paths  .append(self.db_root + "/wc/"  + page_name + ".exr" )
                self._uv_paths  .append(self.db_root + "/uv/"  + page_name + ".exr" )
            names.sort(key=lambda name: ("%04i" % int(name.split("_")[0]) + "%04i" % int(name.split("_")[1].split("-")[0]) + name.split("_")[2].split("-")[0]))  ### 這一步只是想把 list 內容物 的順序 弄得很像 windows 資料夾內的 排序方式比較好找資料，省略也沒關係喔！


class Real_Doc3D(Doc3D):
    def __init__(self, root):
        super(Real_Doc3D, self).__init__(root)
        self.dir_amount = 21
        self.dir_data_amounts = [4999, 5000, 5000, 5000, 5000, 5000, 5000, 2066, 4999, 5000,
                                 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
                                 5000]
        self.dir_data_amounts_acc = np.cumsum(self.dir_data_amounts)



    def _make_fake_some_db(self, dir_key, dst_dir, build_dir_fake_amount=1, sep_dir=True, print_msg=False):
        # self.get_page_names_and_paths()
        '''
        寫的general一些，可以套用到 alb, bm, dmap, img, norm, recon, uv, wc
        dir_key：決定用 alb, bm, dmap, img, norm, recon, uv, wc 的哪一個
        dst_dir：會在那個地方建出 fake_doc3D/alb/, fake_doc3D/bm/, ...
        build_dir_fake_amount：Doc3D總共有21個資料夾，美個資料夾抓出前 多少筆 data 複製出來
        '''
        ### 決定 ord_paths
        if  (dir_key == "img"): ord_paths = self.img_paths
        elif(dir_key == "wc" ): ord_paths = self.wc_paths
        elif(dir_key == "uv" ): ord_paths = self.uv_paths

        ### 決定 副檔名
        if(dir_key in ["wc", "uv", "bm", "norm", "dmap"]): extend_name = ".exr"
        elif(dir_key in ["img", "recon"]): extend_name = ".png"

        ### 定位出 目的地 資料夾
        fake_dst_dir = dst_dir + "/" + dir_key  ### 比如 C:/Users/VAIO_Kong/Desktop/fake_doc3D/wc

        ### 計算出 1~21 號資料夾 的 各個資料夾 lower_bound_index 和 upper_bound_index
        lower_bounds = np.concatenate(([0] , self.dir_data_amounts_acc[:-1]))  ### 定 1~21 號資料夾 的 lower_bound_index，比如：[    0  4999  9999 14999 ... 97064]
        upper_bounds = lower_bounds + build_dir_fake_amount                    ### 定 1~21 號資料夾 的 upper_bound_index，比如：[   10  5009 10009 15009 ... 97074]
        upper_bounds_max = self.dir_data_amounts_acc - 1  ### 最大的upper bound
        if(print_msg): print("lower_bounds:", lower_bounds)
        if(print_msg): print("upper_bounds:", upper_bounds)

        ### 1~21號 資料夾 一一來處理囉！
        for dir_index in tqdm(range(self.dir_amount)):
            dir_lower_b = lower_bounds[dir_index]                                    ### 取出 某號資料夾的 lower_bound
            dir_lower_u = min(upper_bounds[dir_index], upper_bounds_max[dir_index])  ### 取出 某號資料夾的 upper_bound，如果超過 upper_bound_max， 就取upper_bound_max
            dir_name = dir_index + 1                                                 ### 資料夾名是 1~21， 但 index 是 0~20， 所以要 +1

            if(sep_dir): Check_dir_exist_and_build_new_dir(fake_dst_dir + "/" + str(dir_name))    ### 刪掉上次的結果建新的資料夾，要不然比如上次 fake_amount=100， 這次 fake_amount=10， 仍會是100的結果
            else       : Check_dir_exist_and_build        (fake_dst_dir )                         ### 刪掉上次的結果建新的資料夾，要不然比如上次 fake_amount=100， 這次 fake_amount=10， 仍會是100的結果
            for index in range(dir_lower_b, dir_lower_u):
                # print("dir_key:", dir_key)
                ord_data_path = ord_paths[index]                                            ### 取出   ord_data_path
                ### 定位出 dst_data_path
                if(sep_dir): dst_data_path = fake_dst_dir + "/" + self.page_names[index] + extend_name
                else       : dst_data_path = fake_dst_dir + "/" + "%02i-%s" % (int(self.page_names[index].split("/")[0]), self.page_names[index].split("/")[1]) + extend_name
                shutil.copy(ord_data_path, dst_data_path)                                   ### 複製過去囉！
                if(print_msg): print(dir_key, "ord_data_path:", ord_data_path, "copy to")
                if(print_msg): print(dir_key, "dst_data_path:", dst_data_path, "finish")

    def make_fake_img_db(self, dst_dir, build_dir_fake_amount=1, sep_dir=True, print_msg=False):
        self._make_fake_some_db(dir_key="img", dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=print_msg)

    def make_fake_wc_db(self, dst_dir, build_dir_fake_amount=1, sep_dir=True, print_msg=False):
        self._make_fake_some_db(dir_key="wc", dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=print_msg)

    def make_fake_uv_db(self, dst_dir, build_dir_fake_amount=1, sep_dir=True, print_msg=False):
        self._make_fake_some_db(dir_key="uv", dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=print_msg)


    def make_fake_db(self, dst_dir, build_dir_fake_amount=1, sep_dir=True, print_msg=False):
        self.make_fake_img_db(dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=False)
        self.make_fake_wc_db (dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=False)
        self.make_fake_uv_db (dst_dir=dst_dir, build_dir_fake_amount=build_dir_fake_amount, sep_dir=sep_dir, print_msg=False)

class Fake_Doc3D(Doc3D):
    def __init__(self, root):
        super(Fake_Doc3D, self).__init__(root)


### VAIO 電腦
# real_doc3D = Real_Doc3D(root="G:/swat3D")
# fake_doc3D_path = "C:/Users/VAIO_Kong/Desktop/fake_doc3D"

### 127.35
real_doc3D = Real_Doc3D(root="K:/swat3D")

### 127.23 2022/04/11
real_doc3D = Real_Doc3D(root="J:/swat3D")

if(__name__ == "__main__"):
    # fake_doc3D_path = "C:/Users/TKU/Desktop/fake_doc3D"  ### 127.35 忘記什麼時候了
    fake_doc3D_path = "L:/Doc3D_50"  ### 127.23 2022/04/11
    fake_doc3D_merge_path = "L:/Doc3D_50_merge"  ### 127.23 2022/04/11

    real_doc3D.make_fake_db(dst_dir=fake_doc3D_path,       build_dir_fake_amount=50)
    real_doc3D.make_fake_db(dst_dir=fake_doc3D_merge_path, build_dir_fake_amount=50, sep_dir=False)
