from wsdream_helper import wsdream_utility
from wsdream_helper import get_input_args

def main():
    in_args = get_input_args.get_input_args()
    wsdream_utility.dataset_downloader(dir=in_args.dir,url=in_args.url)

if __name__ == "__main__":
    main()