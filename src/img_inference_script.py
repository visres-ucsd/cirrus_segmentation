#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# import sys

# os.chdir('..')
# sys.path.append('src')


# In[31]:


from src.img_inference import process_from_img
from src.utils import save_json_dict
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from keras_segmentation.models.unet import vgg_unet


# In[3]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


rnfl_model = vgg_unet(n_classes=2,  input_height=416, input_width=608)
rnfl_model.load_weights('./weights_9_23_2022/rnfl_onh_cirrus.h5')

ilm_model = vgg_unet(n_classes=2,  input_height=416, input_width=608)
ilm_model.load_weights('./weights_9_23_2022/ilm_onh_cirrus.h5')


# In[30]:


# imgs_to_process = list(Path('data_wd/imgs/').glob('*.img'))
# imgs_to_process


# In[28]:


imgs_to_process = pd.read_csv('imgs_sample_50.csv').files.apply(Path).head(3)
imgs_to_process


# In[29]:


for img_fp in tqdm(imgs_to_process):
    json_dict = process_from_img(str(img_fp), ilm_model, rnfl_model)
    outfp = img_fp.parent.joinpath('test_out_json', f'{json_dict.scan_outname}.json')
    outfp.parent.mkdir(exist_ok=True, parents=True)
    save_json_dict(json_dict, str(outfp))


# ## Create random sample of imgs

# In[14]:


# import pandas as pd
# .sample(70, random_state=123).to_csv('imgs_sample_70.csv', index=False)

