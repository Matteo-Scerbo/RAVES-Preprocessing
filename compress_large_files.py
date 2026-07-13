import os
import gzip

# GitHub blocks files larger than 100 MiB.
github_size_limit = 100 * 2**20
overwrite_existing_compressed = True

for root, dirs, files in os.walk(os.path.join('.', 'example environments')):
    # Check if at least one file in the folder exceeds the limit.
    all_within_limit = True
    for file in files:
        if os.path.getsize(os.path.join(root, file)) > github_size_limit:
            all_within_limit = False
            break

    # If even a single file passes the limit, compress all ART and MoD-ART files.
    if not all_within_limit:
        # Print folder name in order to know what to add in .gitignore
        print(root)
        for file in files:
            # Only compress uncompressed files
            if '.gz' not in file:
                # Only compress ART kernels and MoD-ART data
                if 'ART_kernel_' in file or file in ['MoD-ART.csv', 'MoD-ART extra.csv']:
                    file_path = os.path.join(root, file)
                    compressed_file_path = file_path + '.gz'

                    if not os.path.isfile(compressed_file_path) or overwrite_existing_compressed:
                        # https://stackoverflow.com/a/8156730
                        with open(file_path, 'rb') as f_in, gzip.open(compressed_file_path, 'wb') as f_out:
                            f_out.writelines(f_in)

                    if os.path.getsize(compressed_file_path) > github_size_limit:
                        print('Even the compressed version is too large for file\n\t', file_path)
