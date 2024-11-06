import cv2
import numpy as np
import os

from read_write_model import *

model_dir = '/home/ubuntu/data/gaspar/sparse/0'
image_dir =  '/home/ubuntu/data/gaspar/images'
cameras, images, points3D = read_model(model_dir)
print(cameras)
K = np.array([cameras[1].params[0],0,cameras[1].params[2],cameras[1].params[0],cameras[1].params[1],cameras[1].params[3],0,0,1]).reshape((3,3))

for k,image in images.items():
    R = image.qvec2rotmat()
    t = image.tvec
    depth_map = np.zeros((512,512),dtype=float)
    for id in image.point3D_ids:
        point3d = points3D[id]
        pw = point3d.xyz
        pc = R@pw+t
        d = pc[2]
        u = pc/pc[2]
        x = (K@u).astype(int)
        if x[0]<512 and x[1]<512:
            depth_map[x[1],x[0]] = d
    depth_path = os.path.join(image_dir.replace('images','depth'), image.name.replace('.png','.npy'))
    print(depth_path)
    np.save(depth_path,depth_map)
