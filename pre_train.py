import glob
import os

import tqdm

# specify the directory containing the files
directory = "./abstracts_v2"

# create an empty list to store the contents of each file
file_contents = []

# use the glob module to find all files in the directory
files = glob.glob(os.path.join(directory, "*"))

# loop over each file and read its contents
for file in tqdm.tqdm(files):
    with open(file, "r") as f:
        content = f.read()
        file_contents.append(content)

# print the list of file contents
print(len(file_contents))
