#!/usr/bin/env python
# coding: utf-8

# In[7]:

from astropy.modeling.models import Gaussian1D, Linear1D
import numpy as np
import pandas as pd

def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)



def center_one_trace(kcol, col, fitter, stddev, y_idx, inds, idx_buffer=10):
    model = Gaussian1D(amplitude=col.max(),
                       mean=y_idx, stddev=stddev)

    # idx_narrow = abs(inds - y_idx) < idx_buffer

    # results = fitter(model, inds[idx_narrow], col[idx_narrow])
    results = fitter(model, inds, col)

    return kcol, results, fitter


# In[8]:


def fit_one_slopes(kimg, means, fitter, y_idx, slope_guess=2.0 / 466):
    model = Linear1D(slope=slope_guess, intercept=y_idx)

    inds = np.arange(len(means))
    inds = inds - np.median(inds)

    results = fitter(model, inds, means)

    return kimg, results, fitter


# In[9]:


def cosmic_ray_flag_simple(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_Med = np.median(row)
        row_Std = np.std(row)
        cosmic_rays_[k] += abs(row - row_Med) > n_sig * row_Std
        image_[k][cosmic_rays_[k]] = row_Med

    return image_, cosmic_rays_


def cosmic_ray_flag_rolling(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_rMed = pd.Series(row).rolling(window).median()
        row_rStd = pd.Series(row).rolling(window).std()
        cosmic_rays_[k] += abs(row - row_rMed) > n_sig * row_rStd
        image_[k][cosmic_rays_[k]] = row_rMed[cosmic_rays_[k]]

    return image_, cosmic_rays_


# In[10]:


def aper_table_2_df(aper_phots, aper_widths, aper_heights, n_images):
    info_message(f'Restructuring Aperture Photometry into DataFrames')
    if len(aper_phots) > 1:
        aper_df = aper_phots[0].to_pandas()
        for kimg in aper_phots[1:]:
            aper_df = pd.concat([aper_df, kimg.to_pandas()], ignore_index=True)
    else:
        aper_df = aper_phots.to_pandas()

    photometry_df_ = aper_df.reset_index().drop(['index', 'id'], axis=1)
    mesh_widths, mesh_heights = np.meshgrid(aper_widths, aper_heights)

    mesh_widths = mesh_widths.flatten()
    mesh_heights = mesh_heights.flatten()
    aperture_columns = [colname
                        for colname in photometry_df_.columns
                        if 'aperture_sum_' in colname]

    photometry_df = pd.DataFrame([])
    for colname in aperture_columns:
        aper_id = int(colname.replace('aperture_sum_', ''))
        aper_width_ = mesh_widths[aper_id].astype(int)
        aper_height_ = mesh_heights[aper_id].astype(int)
        newname = f'aperture_sum_{aper_width_}x{aper_height_}'

        photometry_df[newname] = photometry_df_[colname]

    photometry_df['xcenter'] = photometry_df_['xcenter']
    photometry_df['ycenter'] = photometry_df_['ycenter']

    return photometry_df


# In[11]:


def make_mask_cosmic_rays_temporal_simple(val, kcol, krow, n_sig=5):
    val_Med = np.median(val)
    val_Std = np.std(val)
    mask = abs(val - val_Med) > n_sig * val_Std
    return kcol, krow, mask, val_Med


# In[12]:


def check_if_column_exists(existing_photometry_df, new_photometry_df, colname):
    existing_columns = existing_photometry_df.columns

    exists = False
    similar = False
    if colname in existing_columns:
        existing_vec = existing_photometry_df[colname]
        new_vec = new_photometry_df[colname]

        exists = True
        similar = np.allclose(existing_vec, new_vec)

        if similar:
            return exists, similar, colname
        else:
            same_name = []
            for colname in existing_columns:
                if f'colname_{len(same_name)}' in existing_columns:
                    same_name.append(colname)

            return exists, similar, f'colname_{len(same_name)}'
    else:
        return exists, similar, colname


# In[13]:


def rename_file(filename, data_dir='./', base_time=2400000.5,
                format='jd', scale='utc'):

    path_in = os.path.join(data_dir, filename)
    header = fits.getheader(path_in, ext=0)
    time_stamp = 0.5 * (header['EXPSTART'] + header['EXPEND'])
    time_obj = astropy.time.Time(val=time_stamp, val2=base_time,
                                 format=format, scale=scale)

    out_filename = f'{time_obj.isot}_{filename}'
    path_out = os.path.join(data_dir, out_filename)

    os.rename(path_in, path_out)


# In[ ]:
