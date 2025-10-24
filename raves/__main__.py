"""
This lets you run RAVES from the command line, using the following syntax:
```
python -m raves "path/to/your/environment/folder"
```
Assuming you run this command from the root directory of the repository.
Run it with argument `-h` or `--help` to see a list of optional arguments.
For an example, try
```
python -m raves "./example environments/AudioForGames_20_patches"
```
"""
from .cli import main

if __name__ == "__main__":
    main()
