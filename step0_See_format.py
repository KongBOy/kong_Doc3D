from step0_Doc3D_obj  import real_doc3D
from step0_Kong_Doc3D import kong_doc3D

import cv2
import numpy as np
import matplotlib.pyplot as plt

if(__name__ == "__main__"):
    doc3d_uv_paths      = real_doc3D.uv_paths
    doc3d_wc_paths      = real_doc3D.wc_paths
    kong_doc3d_uv_paths = kong_doc3D.uv_npy_paths
    kong_doc3d_wc_paths = kong_doc3D._W_w_M_npy_paths

    for id, _ in enumerate(doc3d_uv_paths):
        doc3d_uv_path      = doc3d_uv_paths[id]
        doc3d_wc_path      = doc3d_wc_paths[id]
        kong_doc3d_uv_path = kong_doc3d_uv_paths[id]
        kong_doc3d_wc_path = kong_doc3d_wc_paths[id]

        doc3d_uv = cv2.imread(doc3d_uv_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        doc3d_wc = cv2.imread(doc3d_wc_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        kong_doc3d_uv = np.load(kong_doc3d_uv_path)
        kong_doc3d_wc = np.load(kong_doc3d_wc_path)

        print("doc3d_uv.shape:", doc3d_uv.shape)
        print("doc3d_wc.shape:", doc3d_wc.shape)
        print("kong_doc3d_uv.shape:", kong_doc3d_uv.shape)
        print("kong_doc3d_wc.shape:", kong_doc3d_wc.shape)

        ### doc3d visualize
        doc3d_uv_ch0 = doc3d_uv[..., 0]  ### Mask
        doc3d_uv_ch1 = doc3d_uv[..., 1]  ### y
        doc3d_uv_ch2 = doc3d_uv[..., 2]  ### x

        doc3d_wc_ch0 = doc3d_wc[..., 0]  ### z
        doc3d_wc_ch1 = doc3d_wc[..., 1]  ### x
        doc3d_wc_ch2 = doc3d_wc[..., 2]  ### y
        doc3d_wc_ch3 = doc3d_wc[..., 3]  ### 全為 1 的東西

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(4 * 4, 4 * 2))
        ax[0, 0].imshow(doc3d_uv_ch0)
        ax[0, 1].imshow(doc3d_uv_ch1)
        ax[0, 2].imshow(doc3d_uv_ch2)
        ax[1, 0].imshow(doc3d_wc_ch0)
        ax[1, 1].imshow(doc3d_wc_ch1)
        ax[1, 2].imshow(doc3d_wc_ch2)
        ax[1, 3].imshow(doc3d_wc_ch3)
        fig.tight_layout()

        ### kong_doc3d visualize
        kong_doc3d_uv_ch0 = kong_doc3d_uv[..., 0]  ### Mask
        kong_doc3d_uv_ch1 = kong_doc3d_uv[..., 1]  ### y
        kong_doc3d_uv_ch2 = kong_doc3d_uv[..., 2]  ### x

        kong_doc3d_wc_ch0 = kong_doc3d_wc[..., 0]  ### z
        kong_doc3d_wc_ch1 = kong_doc3d_wc[..., 1]  ### y
        kong_doc3d_wc_ch2 = kong_doc3d_wc[..., 2]  ### x
        kong_doc3d_wc_ch3 = kong_doc3d_wc[..., 3]  ### 我改成放 Mask

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(4 * 4, 4 * 2))
        ax[0, 0].imshow(kong_doc3d_uv_ch0)
        ax[0, 1].imshow(kong_doc3d_uv_ch1)
        ax[0, 2].imshow(kong_doc3d_uv_ch2)
        ax[1, 0].imshow(kong_doc3d_wc_ch0)
        ax[1, 1].imshow(kong_doc3d_wc_ch1)
        ax[1, 2].imshow(kong_doc3d_wc_ch2)
        ax[1, 3].imshow(kong_doc3d_wc_ch3)
        fig.tight_layout()
        plt.show()
        plt.close()
