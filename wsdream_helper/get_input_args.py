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