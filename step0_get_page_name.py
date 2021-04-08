import os


class Doc3D:
    def __init__(self, root):
        self.db_root = root  ### Doc3D
        self.page_names = []
        self.get_page_name()

    def get_page_name(self):
        train_name_dirs = [self.db_root + "/img/%i" % i for i in range(1, 22)]  ### ["F:/swat3D/1", "F:/swat3D/2", ..., "F:/swat3D/21"]
        for i , name_dir in enumerate(train_name_dirs):
            dir_id = i + 1
            names = os.listdir(name_dir)  ### 1000_1-cp_Page_0179-r8T0001.png  105_3-ny_Page_850-02a0001.png
            names.sort(key=lambda name: ("%04i" % int(name.split("_")[0]) + "%04i" % int(name.split("_")[1].split("-")[0]) + name.split("_")[2].split("-")[0]))
            for name in names:
                self.page_names.append(str(dir_id) + "/" + name[: -4])  ### 去掉副檔名


class Doc3D_train_val_txt:
    @staticmethod
    def write(doc3D, train_split=0.8):
        train_txt_path = doc3D.db_root + "/" + "train.txt"
        val_txt_path   = doc3D.db_root + "/" + "val.txt"
        if os.path.isfile(train_txt_path): os.remove(train_txt_path)
        if os.path.isfile(val_txt_path)  : os.remove(val_txt_path)

        train_num = int(train_split * len(doc3D.page_names))
        with open(train_txt_path, "a") as f:
            for page_name in doc3D.page_names[:train_num]:
                # print(page_name)
                f.write(page_name + "\n")

        with open(val_txt_path, "a") as f:
            for page_name in doc3D.page_names[train_num:]:
                # print(page_name)
                f.write(page_name + "\n")


doc3D = Doc3D(root="D:/swat3D")

if __name__ == "__main__":
    Doc3D_train_val_txt.write(doc3D)
