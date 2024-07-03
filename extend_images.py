import os
import cv2


root="/media/mayue/Data_8T_D/Projects/ZhangXiao/Unet/datasets/val_set/blur"
for file in os.listdir(root):
    img1=cv2.imread(os.path.join(root,file))
    img2=cv2.copyMakeBorder(img1,8,8,8,8,cv2.BORDER_DEFAULT)
    cv2.imwrite(os.path.join(root,file),img2)
