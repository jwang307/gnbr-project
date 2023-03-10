import os
import re
import tarfile
import xml.etree.ElementTree as ET
from ftplib import FTP

import tqdm

# Set up the FTP connection
ftp = FTP('ftp.ncbi.nlm.nih.gov')
ftp.login()

directories = [
    "/pub/pmc/oa_bulk/oa_comm/xml/",
    "/pub/pmc/oa_bulk/oa_noncomm/xml/",
    "/pub/pmc/oa_bulk/oa_other/xml/",
]

# Define a recursive function to extract text from the tag and all its child elements and subchildren at any depth
def extract_text(elem):
    elem_text = elem.text.strip() if elem.text else ''
    for child in elem:
        elem_text += extract_text(child)
    return elem_text

i = 0

for directory in directories:
    # Change to the target directory
    ftp.cwd(directory)

    # Get a list of all the tar files in the directory
    tar_files = [file for file in ftp.nlst() if file.endswith('.tar.gz')]

    # Download each tar file to the local directory
    for file in tqdm.tqdm(tar_files):
        with open("temp.tar.gz", 'wb') as f:
            ftp.retrbinary('RETR {}'.format(file), f.write)

        with tarfile.open('temp.tar.gz', 'r:gz') as tar:
            tar.extractall(path='temp')

        for subdir, dirs, files in os.walk("./temp"):
            for file in files:
                # Check if the file has a .txt extension
                if file.endswith('.xml'):
                    # Open the file and read its contents
                    file_path = os.path.join(subdir, file)

                    tree = ET.parse(file_path)
                    # Get the root element of the XML document
                    root = tree.getroot()

                    # Define the tag name that you want to extract text from
                    tag_name = 'abstract'

                    # Loop through all elements in the XML document with the specified tag and extract text using the recursive function
                    abstract = ""

                    for elem in root.iter(tag_name):
                        elem_text = extract_text(elem)
                        abstract += elem_text

                    with open(f"./abstracts/{i}.txt", "w") as text_file:
                        text_file.write(abstract)
            
                i += 1

        # Delete the temporary directory
        os.system('rm -rf temp')
        # Delete the temporary tar file
        os.system('rm temp.tar.gz')

# Close the FTP connection
ftp.quit()
