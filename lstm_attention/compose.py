import ast
import os
import time

from feature_extraction_clean import roll
import data_utils_compose
from keras.models import model_from_json
import numpy as np
import glob; import os
from os import listdir
import data_utils_train 

train = False
np.set_printoptions(threshold=np.nan) 

mel_dir = 'data/'
composition_dir = 'output/'

old_test = glob.glob("%s*.mid" %(composition_dir))
if len(old_test)>0:
    for i in old_test:
            os.remove(i)

date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
mel_files = glob.glob("%s*.mid" %(mel_dir))    
composition_files = []
for i in range(len(mel_files)):
    composition_files.append(date_and_time + '_%d' %(i+1))

mel_lowest_note = 60

resolution_factor = 12

mel_roll = roll(train)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)
test_data = data_utils_compose.createNetInputs(double_mel_roll, 256)

batch_size = 128
thresh = float(input('Threshold (recommended ~ 0.1):'))

#Load model file
model_dir = 'models/models json/'
model_files = listdir(model_dir)
print("Choose a file for the json model file:")
print("---------------------------------------")
for i, file in enumerate(model_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_number_model = int(input('Type in the number in front of the file you want to choose:')) 
model_file = model_files[file_number_model]
model_path = '%s%s' %(model_dir, model_file)

#Load weights file
weights_dir = 'models/models weights/'
weights_files = listdir(weights_dir)

print("Choose a file for the weights (Model and Weights MUST correspond!):")
print("---------------------------------------")
for i, file in enumerate(weights_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_number_weights = int(input('Type in the number in front of the file you want to choose:')) 
weights_file = weights_files[file_number_weights]
weights_path = '%s%s' %(weights_dir, weights_file)




model = model_from_json(open(model_path).read())
model.load_weights(weights_path)
model.compile(loss='binary_crossentropy', optimizer='adam')

net_output = []
net_roll = []
for i, song in enumerate(test_data):
    net_output.append(model.predict(song))
    net_roll.append(data_utils_compose.NetOutToPianoRoll(net_output[i], threshold=thresh))
    data_utils_compose.createMidiFromPianoRoll(net_roll[i], mel_lowest_note, composition_dir,
                                               composition_files[i], thresh)

orig = glob.glob('data/*.mid')
composed = glob.glob('data/split/*.mid')
for i,j in zip(orig,composed):
    data_utils_compose.merge_left_right(i,j)

