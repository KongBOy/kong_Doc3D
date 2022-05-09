from step0_Doc3D_obj import Doc3D

class Kong_Doc3D(Doc3D):
    def __init__(self, root, use_sep_name=True):
        super(Kong_Doc3D, self).__init__(root)
        self.db_base_name_dir_string = self.db_root + "/0_dis_img/%02i"  ### overwrite 父class
        # self.db_base_name_dir_string = self.db_root + "/2_wc-4_W_w_M_npy/%02i"  ### overwrite 父class， 有些不完整的 Kong_Doc3D 可能會用到， 比如 500GB SSD 容量太小只能裝不完整 Kong_Doc3D

        self.use_sep_name = use_sep_name

        self._dis_img_paths       = None
        self._rec_hope_paths      = None
        self._uv_npy_paths        = None
        self._uv_visual_paths     = None
        self._uv_knpy_paths       = None
        self._wc_npy_paths        = None
        self._wc_2D_visual_paths  = None
        self._wc_3D_visual_paths  = None
        self._W_w_M_npy_paths  = None
        self._W_w_M_knpy_paths = None

    @property
    def dis_img_paths(self):
        if(self._dis_img_paths is None): self.get_doc3d_kinds_of_paths()
        return self._dis_img_paths

    @property
    def rec_hope_paths(self):
        if(self._rec_hope_paths is None): self.get_doc3d_kinds_of_paths()
        return self._rec_hope_paths

    @property
    def uv_npy_paths(self):
        if(self._uv_npy_paths is None): self.get_doc3d_kinds_of_paths()
        return self._uv_npy_paths

    @property
    def uv_visual_paths(self):
        if(self._uv_visual_paths is None): self.get_doc3d_kinds_of_paths()
        return self._uv_visual_paths

    @property
    def uv_knpy_paths(self):
        if(self._uv_knpy_paths is None): self.get_doc3d_kinds_of_paths()
        return self._uv_knpy_paths

    @property
    def wc_npy_paths(self):
        if(self._wc_npy_paths is None): self.get_doc3d_kinds_of_paths()
        return self._wc_npy_paths

    @property
    def wc_2D_visual_paths(self):
        if(self._wc_2D_visual_paths is None): self.get_doc3d_kinds_of_paths()
        return self._wc_2D_visual_paths

    @property
    def wc_3D_visual_paths(self):
        if(self._wc_3D_visual_paths is None): self.get_doc3d_kinds_of_paths()
        return self._wc_3D_visual_paths

    @property
    def W_w_M_npy_paths(self):
        if(self._W_w_M_npy_paths is None): self.get_doc3d_kinds_of_paths()
        return self._W_w_M_npy_paths

    @property
    def W_w_M_knpy_paths(self):
        if(self._W_w_M_knpy_paths is None): self.get_doc3d_kinds_of_paths()
        return self._W_w_M_knpy_paths

    def get_doc3d_kinds_of_paths(self):  ### overwrite 父class
        print("get_doc3d_kinds_of_paths( here~~~~~ should just be used only once!應該只會被用到一次")
        self._get_base_name()  ### reuse 父class

        self._dis_img_paths       = []
        self._rec_hope_paths      = []
        self._uv_npy_paths        = []
        self._uv_visual_paths     = []
        self._uv_knpy_paths       = []
        self._wc_npy_paths        = []
        self._wc_2D_visual_paths  = []
        self._wc_3D_visual_paths  = []
        self._W_w_M_npy_paths  = []
        self._W_w_M_knpy_paths = []

        ### 看看有沒有用 sep_name
        if(self.use_sep_name is True): use_what_names = self.page_names_w_dir_combine_sep
        else:                          use_what_names = self.page_names_w_dir_combine

        for use_what_name in use_what_names:  ### reuse 父class
            self._dis_img_paths       .append(self.db_root + "/0_dis_img/"         + use_what_name + ".png" )
            self._rec_hope_paths      .append(self.db_root + "/0_rec_hope/"        + use_what_name + ".png" )
            self._uv_npy_paths        .append(self.db_root + "/1_uv-1_npy/"        + use_what_name + ".npy" )
            self._uv_visual_paths     .append(self.db_root + "/1_uv-2_visual/"     + use_what_name + ".jpg" )
            self._uv_knpy_paths       .append(self.db_root + "/1_uv-3_knpy/"       + use_what_name + ".knpy" )
            self._wc_npy_paths        .append(self.db_root + "/2_wc-1_npy/"        + use_what_name + ".npy" )
            self._wc_2D_visual_paths  .append(self.db_root + "/2_wc-2_2D_visual/"  + use_what_name + ".jpg" )
            self._wc_3D_visual_paths  .append(self.db_root + "/2_wc-3_3D_visual/"  + use_what_name + ".jpg" )
            self._W_w_M_npy_paths  .append(self.db_root + "/2_wc-4_W_w_M_npy/"  + use_what_name + ".npy" )
            self._W_w_M_knpy_paths .append(self.db_root + "/2_wc-5_W_w_M_knpy/" + use_what_name + ".knpy" )

kong_doc3D = Kong_Doc3D(root=r"E:\data_dir\datasets\type8_blender\kong_doc3d\train")  ### 127.23 2022/04/11

if(__name__ == "__main__"):
    print("dis_img_paths       :", kong_doc3D.dis_img_paths[0])
    print("rec_hope_paths      :", kong_doc3D.rec_hope_paths[0])
    print("uv_npy_paths        :", kong_doc3D.uv_npy_paths[0])
    print("uv_visual_paths     :", kong_doc3D.uv_visual_paths[0])
    print("uv_knpy_paths       :", kong_doc3D.uv_knpy_paths[0])
    print("wc_npy_paths        :", kong_doc3D.wc_npy_paths[0])
    print("wc_2D_visual_paths  :", kong_doc3D.wc_2D_visual_paths[0])
    print("wc_3D_visual_paths  :", kong_doc3D.wc_3D_visual_paths[0])
    print("W_w_M_npy_paths  :", kong_doc3D.W_w_M_npy_paths[0])
    print("W_w_M_knpy_paths :", kong_doc3D.W_w_M_knpy_paths[0])
