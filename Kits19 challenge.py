#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install ipywidgets')
get_ipython().run_line_magic('pip', 'install fastai')
get_ipython().run_line_magic('pip', 'install tqdm')
get_ipython().run_line_magic('pip', 'install pydicom')


# In[2]:


get_ipython().run_line_magic('pip', 'install ipywidgets')


# In[3]:


import os
import glob
import cv2
import imageio
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure

from fastai.basics import *
from fastai.vision.all import*
from fastai.data.transforms import*


# In[2]:


import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import pydicom as dicom
import skimage.transform

from scipy.ndimage import zoom
from skimage import exposure


# In[7]:


get_ipython().run_line_magic('pip', 'install skimage')


# In[3]:


pat_path_1 = '/Users/matteo/Desktop/bio_imaging/dicom_dati_completi_2/33/imaging.nii.gz'


# In[4]:


img_vol = []
for path, _, files in sorted(os.walk(pat_path_1)): 
  for filename in (sorted(files)): 
      if filename.endswith (dcm_ext):
        print (filename)
        img_dcm_std = dicom.dcmread(os.path.join(pat_path_1,filename))

        img = img_dcm_std.pixel_array
        img_vol.append (img)
    
  z_space = img_dcm_std.SpacingBetweenSlices
  x_space = img_dcm_std.PixelSpacing [0]
  y_space = img_dcm_std.PixelSpacing [1]
  vox_dim_1 = (x_space, y_space, z_space)

  img_vol_raw_1 = np.array (img_vol)


# In[5]:


print ('Original image 1 shape: ', img_vol_raw_1.shape)
print ('Voxel dimension image 1: ', vox_dim_1)


# In[ ]:


mid_slice_1 = int(np.round(img_vol_raw_1.shape[0]/2))
print ('Middle slice image 1: ', mid_slice_1)


# In[6]:


fig = plt.figure(figsize=(8,8))
a = fig.add_subplot(1,2,1)
imgplot = plt.imshow(img_vol_raw_1 [mid_slice_1, :, :], cmap = 'gray')
a = fig.add_subplot(1,2,2)

imgplot = plt.imshow(img_vol_raw_2 [mid_slice_2, :, :], cmap = 'gray')


# In[4]:


file_list_input=[]
segmented_file=[]
for dirname, _,filenames in os.walk('/Users/matteo/Desktop/bio_imaging/data'):
    for filename in filenames:
        if 'segmentation' in filename:
            segmented_file.append((dirname,filename))
        else:
                file_list_input.append((dirname, filename))

df_files_imaging=pd.DataFrame(file_list_input,columns=['dirname_input','filename'])
df_files_segmentation=pd.DataFrame(segmented_file,columns=['dirname_segmentation','filename_seg'])


# In[5]:


df=[df_files_imaging,df_files_segmentation]

final_df=pd.concat([df_files_imaging,df_files_segmentation],axis=1)


# In[6]:


final_df.head()


# In[7]:


def read_nii(filepath):
    ct_scan=nib.load(filepath)
    array=ct_scan.get_fdata()
    array=np.rot90(np.array(array),k=1,axes=(0,2))
    
    return(array)


# In[8]:


sample=13
sample_ct=read_nii(final_df.loc[sample,'dirname_input']+"/"+final_df.loc[sample,'filename'])
sample_mask=read_nii(final_df.loc[sample,'dirname_segmentation']+"/"+final_df.loc[sample,'filename_seg'])

print(f'CT Shape:   {sample_ct.shape}\nMask Shape: {sample_mask.shape}')


# In[9]:


def vis_im(array_):
    array=np.rot90(np.array(array_),k=-1)
    return(array)


# In[24]:


dicom_windows=types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom=(200,60)
)

@patch

def windowed(self:Tensor, w, l):
    px=self.clone()
    px_min= l - w//2
    px_max= l + w//2 
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

figure(figsize=(8, 6), dpi=100)
plt.imshow(tensor(vis_im(sample_ct[...,45]).astype(np.float32)).windowed(*dicom_windows.custom), cmap=plt.cm.bone);
plt.show()


# In[25]:


def plot_sample(array_list, color_map = 'nipy_spectral'):
    
    fig=plt.figure(figsize=(20,16), dpi=100)
    
    plt.subplot(1,4,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(tensor(array_list[0].astype(np.float32)).windowed(*dicom_windows.custom), cmap='bone');
    plt.title('Windowed Image')
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.imshow(array_list[0], cmpa='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('KId & Mask')
    plt.axis('off')


# In[12]:


sample=25
sample_slice = tensor(vis_im(sample_ct[...,sample]).astype(np.float32))

plot_sample([vis_im(sample_ct[...,sample]),
            vis_im(sample_mask[...,sample])])


# In[20]:


sample_slice.contiguous().view(-1)


# In[21]:


mask = Image.fromarray(sample_mask[...,sample].astype('uint8'), mode="L")
unique, counts = np.unique(mask, return_counts=True)
print(np.array((unique,counts)).T)


# In[22]:


class TensorCTScan(TensorImageBW): _show_args = {'camp':'bone'}
@patch

def freqhist_bins(self:Tensor, n_bins=100):
    imsd = self.contiguous().view(-1).sort()[0]
    t = torch.cat([tensor[0.001],
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t=(len(imsd)*t).long()
    return imsd[t].unique()
@patch

def hist_scaled(self:Tensor, brks=None):
    if salf.device.type=='cuda':return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0.,1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.,1.)

@patch
def to_nchan(x:Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins] 
    if not isinstance(bins,int) or bins!=0: res.append(x.hist_scaled().clamp(0,1))
    dim = [0,1] [x.dim()==3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch

def to_nchan(x:Tensor, wins, bins=None):
    res = [x.windowed(*wins) for win in wins]
    
    if not isinstance(bins,int) or bins!=0: res.append(x.hist_scaled().clamp(0,1))
    dim = [0,1] [x.dim()==3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch

def save_jpg(x:(Tensor),path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins)*255).byte()
    im = Image.fromarray(x.permute(1,2,0).numpy(), mode=['RGB','CMYK'] [x.shape[0]==4])
    im.save(fn, quality=quality)

sample_slice.save_jpg('test.jpg', [dicom_windows.liver, dicom_windows.custom])

        
        


# In[23]:


_,axs=subplots(1,1)

sample_slice.save_jpg('test.jpg', [dicom_windows.liver,dicom_windows.custom])
show_image(Image.open('test.jpg'), ax=axs[0])
show_image(Image.open('test.jpg'), ax=axs[0])
plt.show()


# In[17]:


GENERATE_JPG_FILES = True

path = Path(".")

os.makedirs('train_images', exist_ok=True)
os.makedirs('train_masks', exist_ok=True)

for ii in tqdm(range(0,len(final_df),3)):
    curr_ct        = read_nii(final_df.loc[ii,'dirname_input']+"/"+final_df.loc[ii,'filename'])
    curr_mask      = read_nii(final_df.loc[ii,'dirname_segmentation']+"/"+final_df.loc[ii,'filename_seg'])
    curr_file_name = str(final_df.loc[ii,'filename']).split('.') [0]
    curr_dim       = curr_ct.shape[2]
    
    for curr_slice in range(0,curr_dim,2):
        data = tensor(vis_im(curr_ct[...,curr_slice]).astype(np.float32))
        mask = Image.fromarray(vis_im(curr_mask[...,curr_slice]).astype('unit8'), mode="L")
        data.save_jpg(f"train_images/{curr_file_name}_slice_{curr_slice}.jpg", [dicom_windows.liver,dicom_windows.custom])
        mask.save(f"train_masks/{curr_file_name}_slice_{curr_slice}_mask.png")
    


# In[26]:


BATCH_SIZE = 2
IMAGE_SIZE = 512

codes = np.array(["background","kidney","tumor"])

def get_x(fname:Path): return fname
def label_func(x): return path/'train_masks'/f'{x.stem}_mask.png'

tfms = [IntToFloatTensor(),Normalise()]

db = DataBlock(block=(ImageBlock(), MaskBlock(codes)),
              batch_tfms=tfms,
              splitter=RandomSplitter(),
              item_tfms=[Resize(IMAGE_SIZE)],
              get_items=get_image_files,
              get_y=label_func)
ds = db.datasets(source=path/'train_images')


# In[29]:


idx = 20
imgs = [ds[idx] [0],ds [idx] [1]]
fig, axs = plt.subplots(1, 2)

for i,ax in enumerate(axs.flatten()):
    ax.axis('off')
    ax.imshow(imgs[i])


# In[30]:


np.unique(array(ds[idx] [1]))


# In[31]:


unique, counts = np.unique(array(ds[idx] [1]), return_counts=True)

print( np.array((unique, counts)).T)


# In[32]:


dls = db.dataloaders(path/'train_images',bs = BATCH_SIZE)
dls.show_batch()


# In[27]:


get_ipython().run_line_magic('pip', 'install torchsummary')


# In[29]:


from torchsummary import summary


# In[34]:


learn = unet_learner(dls,
                    resnet18,
                    loss_func=CrossEntropyLossFlat(axis=1)
                    )


# In[35]:


learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback() )


# In[36]:


learn.show_results()


# In[37]:


interp = SegmentationInterpretation.from_learner(learn)
inter.plot_top_losses(k=50)


# In[ ]:


learn.export(path/f'KI_segmentation)


# In[38]:


if (GENERATE_JPG_FILES) :
    
    tfms = [Resize(IMAGE_SIZE), IntToFloatTensor(), Normalize()]
    learn0 = load_learner(path/f'KI_segmentation',cpu=False)
    learn0.dls.transform = tfms


# In[39]:


def nii_tfm(fn,wins):
    
    test_nii = read_nii(fn)
    curr_dim = test_nii.shape[2]
    slices = []
    
    for curr_slice in range(curr_dim):
        data = tensor(test_nii[...,curr_slice].astype(np.float32))
        data = (data.to_nchan(wins)*255).byte()
        slices.append(TensorImage(data))
        
    return slices


# In[40]:


tst = 20 

test_nii    = read_nii(final_df.loc[tst,'dirname_imput']+"/"+final_df.loc[tst,'filename'])
test_mask   = read_nii(final_df.loc[tst,'dirname_segmentation']+"/"+final_df.loc[tst,'filename_seg'])
print(test_nii.shape)

test_slice_idx = 20

sample_slice = tensor(vis_im(test_nii[...,test_slice_idx]).astype(np.float32))

plot_sample([vis_im(test_nii[...,test_slice_idx]), vis_im(test_mask[...,test_slice_idx])])


# In[ ]:


for test_slice_idx in range(0,92):
    plot_sample([vis_im(test_nii[...,test_slice_idx]), vis_im(test_mask[...,test_slice_idx])])
    


# In[ ]:


def nii_tfm(fin,wins):
    
    test_nii  = read_nii(fn)
    curr_dim  = test_nii.shape[2]
    print(curr_dim)
    slices = []
    
    for curr_slice in range(curr_dim):
        data = tensor(vis_im(test_nii[...,curr_slice]).astype(np.float32))
        data = (data.to_nchan(wins)*225).byte()
        slices.append(TensorImage(data))
        
    return slices
test_files = nii_tfm(final_df.loc[tst,'dirname_input']+"/"+final-df.loc[tst,'filename'],[dicom_windows.liver, dicom_windows.custom])


# In[ ]:


test_dl = learn0.dls.test_dl(test_files)
preds, y =learn0.get_preds(dl=test_dl)

predicted_mask = np.argmax(preds, axis=1)


# In[ ]:


predicted_mask.shape


# In[ ]:


def plot_sample_p(array_list, color_map = 'nipy_spectral'):
    
    fig = plt.figure(figsize=(20,16), dpi=100)
    
    plt.subplot(1,4,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('predicted Image')
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(array_list[1], cmap='bone')
    plt.title('original Image')
    plt.axis('off')


# In[ ]:


predicted_mask[i].shape


# In[ ]:


for i in range(0,92):
    plot_sample_p([predicteds_mask[i],vis_im(test_mask[...,i])])

