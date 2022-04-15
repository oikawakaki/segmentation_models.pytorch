import cv2
import os
import numpy as np
import random
from PIL import Image

h=256
w=256
oh=1280
ow=1280
fd_caz="/home/hua/pys/fullMapDataSet/markedRoad/caz_road/"
fd_cz="/home/hua/pys/fullMapDataSet/markedRoad/cz_road/"
fd_extra="/home/hua/pys/fullMapDataSet/markedRoad/extra/"
fd_hz="/home/hua/pys/fullMapDataSet/markedRoad/hz_road/"

fd_jx="/home/hua/pys/fullMapDataSet/markedRoad/jx_road/"
fd_mas="/home/hua/pys/fullMapDataSet/markedRoad/mas_road/"
fd_nj="/home/hua/pys/fullMapDataSet/markedRoad/nj_road/"
fd_sz="/home/hua/pys/fullMapDataSet/markedRoad/sz_road/"

fd_wh="/home/hua/pys/fullMapDataSet/markedRoad/wh_road/"
fd_wx="/home/hua/pys/fullMapDataSet/markedRoad/wx_road/"
fd_xc="/home/hua/pys/fullMapDataSet/markedRoad/xc_road/"
fd_yz="/home/hua/pys/fullMapDataSet/markedRoad/yz_road/"
fd_zj="/home/hua/pys/fullMapDataSet/markedRoad/zj_road/"

read_arr=[fd_caz, fd_cz,fd_extra,fd_hz,  fd_jx, fd_mas,fd_nj,fd_sz,  fd_wh,fd_wx,fd_xc,fd_yz,fd_zj]

savePlace = '/home/hua/pys/segmentation_models.pytorch/data/triRoad256/'
fnames = ['train', 'valid', 'test']
random.seed(1008)

def cropTo(baseID, read_fold):
    file_count = 0
    full_count = 0
    save_count = 0

    for filename in os.listdir(read_fold):
        if 'b.' in filename:
            mf= os.path.join(read_fold, filename)
            mask = cv2.imread( mf)
            # Lmask = np.array(Image.open(mf).convert('L'))
            assert mask is not None
            img = cv2.imread(os.path.join(read_fold, filename.replace('b.', '.')))
            assert img is not None


            for i in range(100):
                y = i * (h // 4)   #always 64
                for j in range(100):
                    x = j * (w // 4) #always 64
                    if y + h <= oh and x + w <= ow:
                        save_name = 'h' + str(baseID + save_count) + '.png'
                        crop = img[y: y + h, x:x + w]
                        crop_mask = mask[y: y + h, x:x + w]
                        isblack = 255.0 > crop_mask.sum()

                        if (not isblack) or (isblack and 0 == full_count % 3):
                            tt = np.sum(crop_mask, axis=2)
                            crop_mask = np.where(255.0 / 2 < tt, 1, 0)

                            rv = random.random()
                            nameid = 0  # fall into train set
                            if 0.02 > rv:
                                nameid = 2  # 2% into test set
                            elif 0.25 > rv:
                                nameid = 1  # 20% into valid set

                            sfx=os.path.join(savePlace, fnames[nameid] + 'X')
                            cv2.imwrite(os.path.join(sfx, save_name), crop)
                            sfy = os.path.join(savePlace, fnames[nameid] + 'Y')
                            cv2.imwrite(os.path.join(sfy, save_name), crop_mask)

                            save_count = save_count + 1

                        full_count = full_count + 1

            print('file ' + str(file_count) + ' done')
            file_count = file_count + 1
            print('full ' + str(full_count)+'   save ' + str(save_count))

    # print("# full total ", full_count)
    # print("# save total ", save_count)
    return save_count

# v= cropTo(0, mas_in, tar_fold)

baseid=0
for fd in read_arr:
    v = cropTo(baseid, fd)
    baseid=baseid+v
    print(v, baseid)
