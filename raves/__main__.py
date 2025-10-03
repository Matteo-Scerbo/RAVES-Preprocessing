# Lets you run: python -m raves "C:/your/environment/folder/path"
import os
import sys
from .src import compute_ART
from .src import compute_MoDART

def main(argv) -> None:
    if len(argv) > 1:
        if os.path.isdir(argv[1]):
            compute_ART(argv[1])
            compute_MoDART(argv[1])
        else:
            print('Not a valid folder path:\n\t' + argv[1])
    else:
        print('No arguments provided. The first argument should be a valid folder path.')

if __name__ == "__main__":
    main(sys.argv)
