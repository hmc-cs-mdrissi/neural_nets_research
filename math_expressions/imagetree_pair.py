import cv2
import os 
import json
from image_extract import extract_traces_image
from translating_math_trees import make_tree_math
import pickle

root_dir = './CROHME2011_data/'
inkml_train_dir = root_dir + 'CROHME_training/'
inkml_test_dir = root_dir + 'CROHME_testGT/CROHME_testGT/'

training_set =  os.listdir(inkml_train_dir)
test_set = os.listdir(inkml_test_dir)
#print ("TRAINING", len(training_set))
#print ("TEST", len(test_set))

def imagesExtract():
	root_dir = './CROHME2011_data/'
	inkml_train_dir = root_dir + 'CROHME_training/'
	inkml_test_dir = root_dir + 'CROHME_test/CROHME_testGT/'
	print ("hi")
	for inkml_name in os.listdir(inkml_train_dir):
		print (inkml_name)
#path = '/Users/yvenica/Desktop/ICFHR_package/CROHME2011_data/CROHME_testGT/CROHME_testGT/' + 'formulaire050-equation070.inkml'
#canvas = extract_traces_image(path)
#cv2.imwrite( "/Users/yvenica/Desktop/test1.jpg", canvas)

def image_tree_pair(inkml_json_path, tree_json_path, keys_json_path, train_flag):
    """
    Takes in the inkml filename json path, tree json path and a train/test flag.
    The flag helps us decide the root directory of images 
    """
    big_progs = False
    with open(inkml_json_path) as handle:
        inkml_dict = json.loads(handle.read())
    with open(keys_json_path) as f:
        inkml_names = json.loads(f.read())
        
    json_dset = json.load(open(tree_json_path))
    for_progs = [make_tree_math(prog, big_tree=big_progs) for prog in json_dset]

    if train_flag:
        inkml_dir = inkml_train_dir
    else:
        inkml_dir = inkml_test_dir

    # Check dimension of inkml filenames and translated trees 
    print (len(inkml_names), len(for_progs))
    outputL = [] 
    for key_idx in range(len(inkml_names)):
        image_path = inkml_dir+inkml_names[key_idx]
        image = extract_traces_image(image_path)
        tree = for_progs[key_idx]
        outputL.append((image,tree))
    return outputL

train_inkml_json_path = './2011_dataset_train.json'
train_tree_json_path = './2011_parsed_train_dset.json'
train_keys_json_path = './2011_parsed_train_keys.json'

test_inkml_json_path = './2011_dataset_test.json'
test_tree_json_path = './2011_parsed_test_dset.json'
test_keys_json_path = './2011_parsed_test_keys.json'

train_data = image_tree_pair(train_inkml_json_path, train_tree_json_path, train_keys_json_path, True)
test_data = image_tree_pair(test_inkml_json_path, test_tree_json_path, test_keys_json_path, False)


#https://stackoverflow.com/questions/25464295/how-to-pickle-a-list/25465148
with open('train_data_short.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('test_data_short.pkl', 'wb') as f:
    pickle.dump(test_data, f)


def main():
    pass






# # How to unpickle
# with open('train_data.pkl', 'rb') as f:
# 	train_data = pickle.load(f)
# with open('test_data.pkl', 'rb') as f:
# 	test_data = pickle.load(f)
# print (train_data)
# print (len(train_data), "train")
# print (len(test_data), "test")

