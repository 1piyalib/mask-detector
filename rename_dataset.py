from os import listdir
import os
dataset_dir = "dataset2\\with_mask"
filename_key = "-with-mask"

"""
Renames the dataset files to the format -  0-with-mask.jpeg, 1-with-mask.jpeg etc 
"""

index =0
for old_file_name in listdir(dataset_dir):  #loop through file names
    file_split = old_file_name.split(".")
    #construct new file name from old file name
    new_file_name = str(index) + filename_key + "."  + file_split[1]
    #build filename with path
    new_file = os.path.join(dataset_dir, new_file_name)
    old_file = os.path.join(dataset_dir, old_file_name)
    #rename files
    os.rename(old_file, new_file)
    index = index + 1
