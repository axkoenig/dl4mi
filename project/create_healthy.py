import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2

# set parameters
savepath = "data"
seed = 999
np.random.seed(seed) # Reset the seed so all runs are the same
random.seed(seed)
MAXVAL = 255  # Range [0 255]

# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
rsna_datapath = "rsna-pneumonia-detection-challenge"
# get all the normal from here
rsna_csvname = "stage_2_detailed_class_info.csv"
# get all the 1s from here since 1 indicate pneumonia
# found that images that aren't pneunmonia and also not normal are classified as 0s
# rsna_csvname2 = "stage_2_train_labels.csv"
rsna_imgpath = "stage_2_train_images"

# parameters for dataset
train = []
test = []
# we only want to count healhty lung XRays
test_count = 0
train_count = 0

#mapping = dict()
#mapping['healthy'] = 'healthy'

# train/test split
split = 0.1

# to avoid duplicates
patient_imgpath = {}

# add rsna cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge to dataset
csv_normal = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname), nrows=None)
# csv_pneu = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname2), nrows=None)
patients = {'normal': []}

# print(csv_normal.size)
# print(csv_normal.shape)

for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])     

# the file under csv_normal has the labels for normal, lung opacity (indicator for non-COVID pneumonia) and no lung opacity but not normal
print("Length of patients dict after first file")
print(len(patients['normal']))

for key in patients.keys():
    arr = np.array(patients[key])
    print("Array Length:")
    print(len(arr))
    if arr.size == 0:
        continue
    # split by patients; download the .npy files from the repo of Covid-Net
    test_patients = np.load('rsna-pneumonia-detection-challenge/rsna_test_patients_{}.npy'.format(key))
  
    print("test patients has size:")
    print(len(test_patients))

    for patient in arr:
        # To avoid duplicates      
        if patient not in patient_imgpath:
            patient_imgpath[patient] = patient
        else:
            continue  # skip since image has already been written
        
        # dcmread is the main function to read and parse DICOM files
        dataset = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
        pixel_array_numpy = dataset.pixel_array # convert image to a numpy array
        imgname = patient + '.png'
        # rsna_test_patients_normal.npy contains all test cases
        if patient in test_patients:
            print(os.path.join(savepath, 'test', imgname))
            cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
            test.append([patient, imgname, key])
            test_count += 1
            print(test_count)
        else:
            cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
            train.append([patient, imgname, key])
            train_count += 1
            print(train_count)

print('test count: ', test_count)
print('train count: ', train_count)        

print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))

# export to train and test txt
# format as patientid, filename, label, separated by a space
train_file = open("train_split_v3.txt","a") # mode a: open for writing, appending to the end of the file if it exists
for sample in train:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()

test_file = open("test_split_v3.txt", "a")
for sample in test:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)

test_file.close()