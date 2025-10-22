"""
This lets you run RAVES from the command line, as
```
python -m raves "C:/your/environment/folder/path"
```
Assuming you run this command from the root directory of the repository.
For an example, try
```
python -m raves "./example environments/AudioForGames_fewest_patches"
```
"""
from .cli import main

if __name__ == "__main__":
    main()
