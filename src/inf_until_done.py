import pandas as pd
import os
from pathlib import Path

def all_masks_compelte(path):
    if not Path(path).exists():
        return False
    masks = pd.Series([str(i) for i in Path(path).rglob('*.jpg')])
    masks = masks[~masks.str.contains('ipynb')]
    scans = masks.apply(lambda p: Path(p).parent.name)
    return (scans.value_counts() == 400).sum() == 1964
    # masks_df = masks.apply(lambda x: str(x).split('/'))
    # masks_df = pd.DataFrame(masks_df.to_list(), columns=['model', 'group', 'pt_id', 'scan', 'png'])
    # return (masks_df[masks_df.other.isna()].scan.value_counts() == 400).sum() == 1964

while not all_masks_compelte('RNFL_masks/'):
    os.system('~/miniconda3/envs/seg-model-env/bin/python inference.py CirrusPNGs/ RNFL')
    
while not all_masks_compelte('ILM_masks/'):
    os.system('~/miniconda3/envs/seg-model-env/bin/python inference.py CirrusPNGs/ ILM')
    
os.system('~/miniconda3/envs/seg-model-env/bin/python post_process.py')

os.system('zip -r ILM_masks.zip ILM_masks')
os.system('zip -r RNFL_masks.zip RNFL_masks')

os.system('aws s3 cp ILM_masks.zip s3://ruben-aws-test/sftp-user/')
os.system('aws s3 cp RNFL_masks.zip s3://ruben-aws-test/sftp-user/')
os.system('aws s3 cp RNFL_thickness.csv s3://ruben-aws-test/sftp-user/')
