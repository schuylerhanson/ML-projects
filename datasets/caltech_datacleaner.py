import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%% Read GT file, get face coords and use these for cropping

gt_file = '/mnt/c/Users/schuy/Documents/datasets/caltech_faces_gt.txt'
with open(gt_file) as f:
    lines = f.readlines()

lines = [x.split(' ') for x in lines]
f_dict = {}
for line in lines:
    f_dict[line[0]] = [float(x) for x in line[1:-1]]

#%% Read in imgs, crop

fdir = '/mnt/c/Users/schuy/Documents/datasets/caltech_10k_faces'
files = [os.path.join(fdir,x) for x in os.listdir(fdir)]
fnames = [os.path.basename(x) for x in files]


# Test over 5 imgs
N=5
imgs = [np.array(Image.open(x), order='F') for x in files[:N]]
fnames = fnames[:N]

coords = [f_dict[x] for x in fnames]
left_eyes = [(x[0], x[1]) for x in coords]
right_eyes = [(x[2], x[3]) for x in coords]


fdir_out = '/mnt/c/Users/schuy/Documents/datasets/caltech_sample'
if not os.path.exists(fdir_out):
    os.makedirs(fdir_out)
for idx,img in enumerate(imgs):
    fig,ax = plt.subplots()
    left_x = '{:.2f}'.format(left_eyes[idx][0])
    left_y = '{:.2f}'.format(left_eyes[idx][1])
    ax.imshow(img); ax.set_title('({},{})'.format(left_x,left_y))
    fname_x = os.path.join(fdir_out, '{}.png'.format(fnames[idx].split('.')[0]))
    fig.savefig(fname_x)


dist = lambda x1,x2: ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)**.5
phi_calc =  lambda x1,x2: np.arctan((x2[1] - x1[1]/(x2[0] - x1[0])))
M_rot = lambda phi: np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


distances = [dist(x1,x2) for x1,x2 in zip(left_eyes,right_eyes)]
phis = [phi_calc(x1,x2) for x1,x2 in zip(left_eyes,right_eyes)]

x_c = [img.shape[0]/2 for img in imgs]
y_c = [img.shape[1]/2 for img in imgs]
#print(x_c, y_c)

IMG_c = [np.array([[xc, 0], [0, yc]]) for xc,yc in zip(x_c,y_c)]
IMG_M = [M_rot(phi) for phi in phis]

LEFT_x = [np.array([[x[0], 0],[0 ,x[1]]]) for x in left_eyes]
LEFT_c = [left_x-img_c for left_x,img_c in zip(LEFT_x,IMG_c)]
RIGHT_x = [np.array([[x[0], 0],[0 ,x[1]]]) for x in right_eyes]
RIGHT_c = [right_x-img_c for right_x,img_c in zip(RIGHT_x,IMG_c)]

#IMG_r_left = [img_c + img_m@left_c for img_c,img_m,left_c in zip(IMG_c,IMG_M,LEFT_c)]
#IMG_r_right = [img_c + img_m@right_c for img_c,img_m,right_c in zip(IMG_c,IMG_M,RIGHT_c)]

get_transformed_vectors = lambda x: np.array([[np.sum(x, axis=0)[0], 0], [0, np.sum(x, axis=0)[1]]])

IMG_r_left = [img_c + get_transformed_vectors(img_m@left_c) for img_c,img_m,left_c in zip(IMG_c,IMG_M,LEFT_c)]
IMG_r_right = [img_c + get_transformed_vectors(img_m@right_c) for img_c,img_m,right_c in zip(IMG_c,IMG_M,RIGHT_c)]

IMG_r_left_0 = IMG_r_left[0]
IMG_r_right_0 = IMG_r_right[0]

IMG_right_T = IMG_r_right_0.T
IMG_left_T = IMG_r_left_0.T


#print(IMG_r_left_0, '\n', IMG_left_T)

gamma = 0.65
get_GAMMA_ul = lambda gamma,dist: np.array([[gamma*dist, 0], [0, gamma*dist]])
get_GAMMA_lr = lambda gamma,dist: np.array([[gamma*dist, 0], [0, (1+gamma)*dist]])

GAMMA_ul = [get_GAMMA_ul(gamma,dist) for dist in distances]
GAMMA_lr = [get_GAMMA_lr(gamma,dist) for dist in distances]

XY_ul = [img_r_left-gamma_ul for img_r_left,gamma_ul in zip(IMG_r_left, GAMMA_ul)]
XY_lr = [img_r_right-gamma_lr for img_r_right,gamma_lr in zip(IMG_r_right, GAMMA_lr)]

print('XY_ul[0]:\n', XY_ul[0])
print('XY_lr[0]:\n', XY_lr[0])

XY_ul_T = [xy_ul.T for xy_ul in XY_ul]
XY_lr_T = [xy_lr.T for xy_lr in XY_lr]

#xy_0 = XY_ul_T[0]
#print('xy_0:', '\n', np.array((xy_0[0], xy_0[1])))


heights = [np.array([np.array([0, 0]), np.array(xy_ul_T[1]-xy_lr_T[1])]).T for xy_ul_T,xy_lr_T in zip(XY_ul_T,XY_lr_T)]
widths = [np.array([np.array(xy_lr_T[0]-xy_ul_T[0]), np.array([0, 0])]).T for xy_ul_T,xy_lr_T in zip(XY_ul_T,XY_lr_T)]

XY_ll = [np.round(xy_ul-height) for xy_ul,height in zip(XY_ul, heights)]

heights_max = [np.round(np.max(height)) for height in heights]
widths_max = [np.round(np.max(width)) for width in widths]

rect_patches = [patches.Rectangle((xy_ll[0,0],xy_ll[1,1]), width, height) for xy_ll,width,height in zip(XY_ll,widths_max,heights_max)]

print(rect_patches)

fig,axs = plt.subplots(2,2,figsize=(12,12))
for idx,ax in enumerate(axs.flatten()):
    ax.imshow(imgs[idx])
    print(XY_ll[idx], widths[idx], heights[idx])
    ax.add_patch(rect_patches[idx])
    ax.set_title(fnames[idx])

fout = os.path.join(os.getcwd(),'alignment_scheme.png')
fig.savefig(fout)


'''
print('IMG_c:', IMG_c[0], '\n')
print('LEFT_x:', LEFT_x[0], '\n')
print('IMG_m:', IMG_M[0], '\n')
print('LEFT_c:', LEFT_c[0], '\n')
print('IMG_r_left:', IMG_r_left[0], '\n')
#print('IMG_m@LEFT_c:', IMG_r_left[0]-IMG_c[0], '\n')

print('----------------', '\n')

print('RIGHT_x:', RIGHT_x[0], '\n')
print('RIGHT_c:', RIGHT_c[0], '\n')
print('IMG_r_right:', IMG_r_right[0], '\n')
#print('IMG_m@RIGHT_c:', IMG_r_right[0]-IMG_c[0], '\n')


#print(distances)
#print(phis)
#print(fnames)
'''
