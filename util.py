# cloud
from google.cloud import bigquery, storage

# essentials
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

# direcories
from os import listdir
from os.path import isfile, join

# querries only front views
def querry_data(bq_client):
    
    query = """
        SELECT *
        FROM `hackathon-2021-305208.common_referential.metadata_images`,
        unnest(arbonodes) a,
        unnest(location) l
        WHERE a.is_primary_link='true' and img_angle in ('1','11','31')
    """
    query_job = bq_client.query(
        query,
        location="EU",
    )

    return query_job.to_dataframe()

# drops the skewed lines to keep only one primary link per product
# drops the null values for the target value
def clean_data(x):
    to_keep = ['img_loc', 'nodeid3', 'level3']
    # either keep all the views
    data = x.drop_duplicates(['barcode', 'img_loc'], keep=False)[to_keep]
    # or keep only the first view for each product
#     data = x.drop_duplicates(['barcode'], keep='first')[to_keep]
    data = data.dropna()

    data = data.rename(columns={"img_loc":"filename", "nodeid3":"label", 'level3': 'level'})
    return data


## loading all images in a folder
def load_images_from_folder(folder):
    images = []
    for filename in listdir(folder):
        img = cv2.imread(join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def show_predictions_for(x_test, y_pred, nrow=8, ncol=5, start=0, val=False, filenames=None):
    
    fig, ax_array = plt.subplots(nrow, ncol,squeeze=False, figsize=(15,15))
    for i,ax_row in enumerate(ax_array):
         for j,axes in enumerate(ax_row):
            if x_test is not None and val ==False:
                im = cv2.cvtColor(x_test[i*nrow + j + start], cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(filenames[i*nrow + j + start])
                im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes.set_title(y_pred[i*nrow + j + start])
            axes.imshow(im)
            
    plt.tight_layout()
    plt.show()
    
    
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def clean_loc(s):
    return remove_prefix(s, "gs://datacamp-images/")

# s: like "gs://datacamp-images/images/image_330713993682..."
def get_image(s, bucket):
    blob = bucket.get_blob(clean_loc(s))
    image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
    bgr_im = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)

def get_image_for_index(index):
    loc = df['img_loc'].loc[index]
    return get_image(loc)

def show_product_views(barcode, df, bucket):
    product_views = df.loc[df['barcode'] == barcode]
    for i, view in product_views.iterrows():
        im = get_image(view['img_loc'], bucket)
        print(view['img_angle'])
        print(im.shape)
        plt.imshow(im)
        plt.show()
        
        

def show_images_for(df, bucket, nrow=8, ncol=5, start=0):
    
    fig, ax_array = plt.subplots(nrow, ncol,squeeze=False, figsize=(15,15))
    for i,ax_row in enumerate(ax_array):
         for j,axes in enumerate(ax_row):
            ind = i * nrow + j
            view = df.iloc[ind]
            im = get_image(view['img_loc'], bucket)

            axes.set_title('Barcode: ' + view['barcode'])
            axes.imshow(im)
            
    plt.tight_layout()
    plt.show()
    
# mapping between labels and classes for encoding decoding purposes
def encode_map(x, labels):
    return {i:j for i,j in zip(np.unique(x['label']), range(labels.nunique()))}

def decode_map(x, labels):
    return {j:i for i,j in zip(np.unique(x['label']), range(labels.nunique()))}

# the one hot encoding of the labels
def one_hot_encoding(data, labels, labels_encode):
    y_one_hot = np.zeros((data.shape[0], labels.nunique()))

    for label in range(data.shape[0]):
        y_one_hot[label, labels_encode[data.iloc[label]['label']]] = 1
    
    return y_one_hot

# retrieve the labels from the one encoded values 
def one_hot_decoding(y_one_hot, labels_decode):
    
    y_reconstruct = np.argmax(y_one_hot, axis=1)

    for i in range(y_reconstruct.shape[0]):
        y_reconstruct[i] = labels_decode[y_reconstruct[i]]
        
    return y_reconstruct