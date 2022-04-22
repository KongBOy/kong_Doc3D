import os

class Doc3D_train_val_txt:
    '''
    train/val.txt 這是給 DewarpNet 用的喔～
    '''
    @staticmethod
    def write(doc3D, train_split=0.8):
        train_txt_path = doc3D.db_root + "/" + "train.txt"
        val_txt_path   = doc3D.db_root + "/" + "val.txt"
        print("train_txt_path:", train_txt_path)
        if os.path.isfile(train_txt_path): os.remove(train_txt_path)
        if os.path.isfile(val_txt_path)  : os.remove(val_txt_path)

        train_num = int(train_split * len(doc3D.page_names_w_dir))
        with open(train_txt_path, "a") as f:
            for page_name_w_dir in doc3D.page_names_w_dir[:train_num]:
                # print(page_name_w_dir)
                f.write(page_name_w_dir + "\n")

        with open(val_txt_path, "a") as f:
            for page_name_w_dir in doc3D.page_names_w_dir[train_num:]:
                # print(page_name_w_dir)
                f.write(page_name_w_dir + "\n")


if __name__ == "__main__":
    from step0_Doc3D_obj import real_doc3D
    Doc3D_train_val_txt.write(real_doc3D)
