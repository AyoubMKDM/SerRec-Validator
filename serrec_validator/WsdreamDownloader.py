import urllib.request as urlopener
import shutil
from zipfile import ZipFile
import os
import argparse
from urllib.error import URLError, HTTPError

def get_input_args():
    """
    Retrieves and parses the 2 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these 2 command line arguments. If 
    the user fails to provide some or all of the 2 arguments, then the default 
    values are used for the missing arguments.
    
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


def check_command_line_arguments(in_args):
    """
    Validates the command-line arguments.

    Parameters:
    in_args (argparse.Namespace): Command-line arguments

    Returns:
    bool: Returns True if arguments are valid, False otherwise
    """
    # Check if the directory path is valid (could be an existing directory or path to be created)
    if not isinstance(in_args.dir, str):
        print("Error: The directory path should be a string.")
        return False
    if not isinstance(in_args.url, str) or not in_args.url.startswith("http"):
        print("Error: The URL is invalid.")
        return False
    
    # TODO add other checks
    return True


def download(dir: str ='wsdream_dataset1', url : str ="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1") -> str:
    """
    Download and extract the WS-DREAM dataset from the given URL.

    Parameters:
    dir (str): The directory where the dataset will be stored. Default is "./wsdream_dataset1".
    url (str): The URL of the downloadable zip file.

    Returns:
    str: The directory where the dataset is stored after downloading and extraction.

    Example:
    >>> dataset_downloader('my_data_folder')
    """

    try:
        # Extract filename from the URL
        file_name = url.split('/')[-1]
        
        # Open URL and retrieve file size
        print(f"Attempting to download from {url} ...")
        page = urlopener.urlopen(url)
        meta = page.info()
        file_size = int(meta.get("Content-Length")[0])

        print(f"Downloading: {file_name} ({file_size} bytes)")
        urlopener.urlretrieve(url, file_name)

        # Unzip the downloaded file
        print("Unzipping dataset files...")
        with ZipFile(file_name, 'r') as archive:
            # Ensure the target extraction directory exists
            extraction_path = os.path.join(os.getcwd(), dir)
            os.makedirs(extraction_path, exist_ok=True)

            # Extract and copy files
            for name in archive.namelist():
                filename = os.path.basename(name)
                if filename:  # Skip directories
                    print(f"Extracting: {filename}")
                    source = archive.open(name)
                    target_path = os.path.join(extraction_path, filename)
                    with open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

        # Remove the zip file after extraction
        os.remove(file_name)

        print('==============================================')
        print('Download and extraction complete!\n')
        return dir

    except HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
    except URLError as e:
        print(f"URL Error: {e.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def main():
    # Retrieve command-line arguments
    in_args = get_input_args()

    # Check validity of the command-line arguments
    if not check_command_line_arguments(in_args):
        print("Invalid arguments provided. Exiting.")
        return

    # Call the download function with the parsed arguments
    result_dir = download(dir=in_args.dir, url=in_args.url)
    if result_dir:
        print(f"Dataset stored in: {result_dir}")
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()