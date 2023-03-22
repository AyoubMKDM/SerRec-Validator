import urllib.request as urlopener
import shutil
from zipfile import ZipFile
import os
# from .get_input_args import get_input_args
import argparse

def get_input_args():
    """
    Retrieves and parses the 2 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 2 command line arguments. If 
    the user fails to provide some or all of the 2 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Dataset folder as --dir with default value 'dataset'
      2. link o the repository containing the dataset as --url with default value 'https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    Parse = argparse.ArgumentParser(description="Process the user inputs")
    Parse.add_argument("--dir", default='dataset', help='Path to the folder where the data is stored. If not available it will be created.', type=str)
    Parse.add_argument("--url", default='https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1',
                         help='Link of the dataset to be downloaded', type=str)
    
    return Parse.parse_args()


def check_command_line_arguments(in_arg):
    # TODO impliment this
    pass

def dataset_downloader(dir=None, url="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1"):
    """
    Retrieve and download the WS-DREAM dataset as a zip file from the specified URL, 
    then extract the content from the zip file and delete the latter. This function 
    will add four files (rtMatrix.txt, tpMatrix.txt, userlist.txt, wslist.txt) to the 
    specified directory or the current directory if no directory is specified.

    Parameters:
    dir (str): The directory where the dataset will be stored. Default is the current directory.
    url (str): The URL of the downloadable zip file. The URL can be modified to download the 
    WS-DREAM first dataset in case of a broken link or changing in the repository.
    The default is 'https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1'.

    Returns:
    None

    Example:
    >>> dataset_downloader('my_data_folder')
    """
    # TODO make this return the file name
    # TODO Handle http and url exceptions .. Use https://docs.python.org/3/library/urllib.error.html#urllib.error.URLError 
    file_name = url.split('/')[-1]
    page = urlopener.urlopen(url)
    meta = page.info()
    file_size = int(meta.get("Content-Length")[0])
    print ("Downloading: %s (%s bytes)" % (file_name, file_size))
    urlopener.urlretrieve(url, file_name)
    print ('Unzip data files...')
    with ZipFile(file_name, 'r') as archive:
        for name in archive.namelist():
            filename = os.path.basename(name)
            # skip directories
            if not filename:
                continue
            # copy file (taken from zipfile's extract)
            print (filename)
            source = archive.open(name)
            if(dir is None):
                target = open(filename, "wb")
            else:
                extraction_path = os.path.join(os.getcwd(),dir)
                if not os.path.exists(extraction_path):
                    print(f"Creating '{dir}/' folder ...")
                    os.mkdir(dir)
                target = open(os.path.join(extraction_path,filename),'wb')
            with source, target:
                shutil.copyfileobj(source, target)

    os.remove(file_name)
    print('==============================================')
    print('Downloading data done!\n')
    pass

def main():
    in_args = get_input_args()
    dataset_downloader(dir=in_args.dir,url=in_args.url)

if __name__ == "__main__":
    main()