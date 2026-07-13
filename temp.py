import os
import csv

material_folder = os.path.join('..', '..', 'BRAS', '3 Surface descriptions', '_csv', 'fitted_estimates')

started = False
absorptions = dict()
scatterings = dict()

for root, dirs, files in os.walk(material_folder):
    for file in files:
        with open(os.path.join(root, file),
                  mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        
            frequencies = next(reader, None)
            
            if not started:
                print('Frequencies, ' + ', '.join(frequencies))
                started = True
            
            print(file[4:-4] + ', ' + ', '.join(next(reader, None)))
            print(file[4:-4] + ', ' + ', '.join(next(reader, None)))
