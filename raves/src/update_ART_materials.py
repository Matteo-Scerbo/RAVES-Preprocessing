import os
import sys

def main(path) -> None:
    if os.path.isdir(path):
        print('I am `update_ART_materials` and I will process the environment named ' + path.split('/')[-1])
    else:
        print('Not a valid folder path:\n\t' + path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print('No arguments provided. The first argument should be a valid folder path.')
