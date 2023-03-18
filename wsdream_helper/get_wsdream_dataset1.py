from .wsdream_utility import dataset_downloader
from .get_input_args import get_input_args

# TODO fix the bug 
# from .wsdream_utility import dataset_downloader
# ImportError: attempted relative import with no known parent package
def main():
    in_args = get_input_args()
    dataset_downloader(dir=in_args.dir,url=in_args.url)

if __name__ == "__main__":
    main()