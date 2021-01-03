import os
import cv2

form = '3_z_3'
fn = f"data_{form}"
vn = f'{form}.mp4'
fp = os.path.join(os.path.abspath(""), fn)

fps = 9
size = (600, 600)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(vn, fourcc, fps, size)

pic_name = os.listdir(fp)
for i, j in enumerate(pic_name):
    try:
        print(int(j.split('.')[0]))
    except ValueError as v:
        pic_name.remove(j)
        print(i, j)
pic_name.sort(key=lambda x: int(x.split('.')[0]))
for p in pic_name:
    pp = os.path.join(fp, p)
    frame = cv2.imread(pp)
    videoWriter.write(frame)

videoWriter.release()


