# Preprocessing script to ingest raw data from the physician labels
import pandas as pd
import numpy as np

# File paths to be changed
# data path and sheet name for the physician labels
# replace with the excel file name
DATA_FILE_PATH = '../data/physician_labels/Labels_TrainImages_1-200 - B - LPL done20250209.xlsx'
# replace with the excel sheet name we want to translate the labels from
DATA_SHEET_NAME = 'Labels_TrainImages_1'

# file path for the translation file for features and labels
TRANSLATION_FILE = '../data/translation/Tongue Features - 4.xlsx'
# sheet name with the translations
TRANSLATION_SHEET_NAME = 'Translated'

# file names of images to remove
IMAGES_TO_REMOVE = []

# export preprocessed file path
EXPORT_PREPROCESSED_PATH = '../data/processed/physician_1.csv'



# helper functions to check integrity of data
def check_features_and_labels(features, labels):
    '''
    Assertions to check integrity of data for the features and labels. This is to ensure proper parsing of features and labels.
    '''
    # check shape and no null values
    assert features.isnull().sum().sum() == 0, "Features dataframe contains null values"
    assert features.shape == (25, 2), "Features dataframe has incorrect shape"

    assert labels.isnull().sum().sum() == 0, "Labels dataframe contains null values"
    assert labels.shape == (24, 2), "Labels dataframe has incorrect shape"

def check_data(df):
    '''
    Assertions to check integrity of physician labelled data. This is to ensure proper parsing of data.
    '''
    assert df.isnull().sum().sum() == 0, "Physician labelled data contains null values"
    assert df.shape == (200, 26), "Physician labelled data has incorrect shape"



# Read in data
# read in the features and their translations we are concerned with (the column names)
features = pd.read_excel(
    io=TRANSLATION_FILE,
    sheet_name=TRANSLATION_SHEET_NAME,
    engine='openpyxl',
    usecols='A:B',
    nrows=25
)

# read in the labels and their translations we are concerned with (the values for all features)
labels = pd.read_excel(
    io=TRANSLATION_FILE,
    sheet_name=TRANSLATION_SHEET_NAME,
    engine='openpyxl',
    usecols='D:E',
    nrows=24
)

# read in the data
df = pd.read_excel(
    io=DATA_FILE_PATH,
    sheet_name=DATA_SHEET_NAME,
    engine='openpyxl',
    usecols='A:Z',
    nrows=200
)

# Preliminary check on whether the features, labels, and data are of correct size
check_features_and_labels(features, labels)
check_data(df)

# Perform the translation of data in the dataframe

feature_dict = features.set_index('Column_Chinese')['Column_English'].to_dict()
# add a dummy key for image_name so that it does not map to nan
feature_dict['image_name'] = 'image_name'

label_dict = labels.set_index('Values_Chinese')['Values_English'].to_dict()

# translate column names and labels
df.columns = df.columns.map(feature_dict)

# loop to loop through all columns and apply translation
for c in df.columns:
    # if it is image name, do not apply any translation
    if c != 'image_name':
        # translate
        df[c] = df[c].map(label_dict)

# TODO: Ignore the other images we do not want, drop them
# need to discuss with Yan Chun the rules on what to remove

# Remove hashtags and special formatting of image files
df['image_name'] = df['image_name'].str.replace(r'\d*#', '', regex=True)

# check if data is ok before exporting
check_data(df)

# Export the processed csv file
df.to_csv(EXPORT_PREPROCESSED_PATH, index=False)
print("Successfully exported translated labels")