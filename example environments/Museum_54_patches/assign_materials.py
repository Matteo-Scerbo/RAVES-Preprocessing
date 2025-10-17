reference_absorptions = dict()
with open('../assorted_absorptions.csv', 'r') as file:
    for line in file:
        tokens = line.replace('\n', '').split(', ')
        reference_absorptions[tokens[0]] = tokens[1:]

reference_scatterings = dict()
with open('../ODEON_recommended_scatterings.csv', 'r') as file:
    for line in file:
        tokens = line.replace('\n', '').split(', ')
        reference_scatterings[tokens[0]] = tokens[1:]

"""
Absorption materials (frequencies: 250.0, 500.0, 1000.0, 2000.0, 4000.0):
['Basalt15', 'Basalt30', 'Basalt46', 'Basalt15Fine',
 'Brick', 'BrickPainted',
 'Concrete', 'ConcreteBlock', 'ConcreteBlockPainted', 'ConcreteBlockPlastered', 'ConcreteFloor', 'Marble',
 'LinoleumOnConcrete', 'WoodBoardsOnConcrete', 'CarpetThinOnConcrete', 'CarpetHeavyOnConcrete',
 'CarpetHeavyOnFoamRubber', 'CarpetHeavyLatexBackingOnFoamRubber', 'WoolLoopCarpet24', 'WoolLoopCarpet64', 'WoolLoopCarpet95', 'LoopPileCarpet7', 'LoopPileCarpet7OnPad', 'LoopPileCarpet14OnPad',
 'CurtainsLight', 'CurtainsMedium', 'CurtainsHeavy',
 'GlassWindow', 'GlassHeavyPlate', 'GlassLargePane', 'GlassSmallPane',
 'GypsumBoard', 'GypsumPlasterBoard', 'PlasterCeiling30', 'PlasterCeiling60', 'PlasterOnBrick', 'PlasterOnLath', 'AcousticalPlaster', 'Plasterboard12OnStuds', 'PlasterboardCeilingSuspendedGrid',
 'WoodenDoor', 'WoodFloor', 'WoodParquetOnConcrete', 'WoodCeiling28', 'WoodSideWalls12', 'WoodSidewalls20', 'WoodFloor33OnSleepersOverConcrete', 'WoodFloor27OverAirspace', 'Wood19OverFibreglassOnConcrete', 'WoodenTongueGrooveCeiling', 'Plywood3OverAirspace32', 'Plywood3OverAirspace57', 'Plywood5OverAirspace50', 'Plywood6', 'Plywood10', 'Plywood19',
 'MetalDeck25Batts', 'MetalDeck75Batts',
 'DryAsphalt', 'WetAsphalt', 'DirtyAsphalt', 'WaterSurface',
 'AudienceHeavyOccupied', 'AudienceMediumOccupied', 'AudienceLightOccupied', 'AudienceHeavyUnoccupied', 'AudienceMediumUnoccupied', 'AudienceLightUnoccupied',
 'Full_absorption'
]
Scattering materials (frequency independent; min and max provided):
['Audience', 'Bookshelf', 'Brickwork_open', 'Brickwork_filled', 'Generic_rough', 'Generic_smooth', 'Smooth_painted_concrete']
"""

with open('./materials.csv', 'w') as file:
    file.write('Frequencies, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0\n')

    all_rooms = ['Outdoors', 'Large_exhibit', 'Small_exhibit', 'Ballroom', 'Hallway', 'Meeting_room', 'Cinema', 'Restroom']

    for room in all_rooms:
        all_walls = ['Floor', 'Ceiling', 'West', 'East', 'North', 'South']
        if room == 'Meeting_room':
            all_walls += ['SouthEast', 'NorthEast', 'NorthWest', 'SouthWest']
        elif room == 'Cinema':
            all_walls += ['Cinema_WestL', 'Cinema_NorthL']

        for wall in all_walls:
            raves_material = room + '_' + wall

            if room == 'Outdoors':
                if wall == 'Floor':
                    abso_material = 'DirtyAsphalt'
                elif wall == 'North':
                    abso_material = 'Brick'
                else:
                    abso_material = 'Full_absorption'
            elif room == 'Large_exhibit':
                if wall == 'Floor':
                    abso_material = 'ConcreteFloor'
                else:
                    abso_material = 'ConcreteBlockPlastered'
            elif room == 'Small_exhibit':
                if wall == 'Floor':
                    abso_material = 'ConcreteFloor'
                else:
                    abso_material = 'BrickPainted'
            elif room == 'Ballroom':
                if wall == 'Floor':
                    abso_material = 'Marble'
                elif wall == 'Ceiling':
                    abso_material = 'PlasterCeiling30'
                elif wall == 'East':
                    abso_material = 'CurtainsLight'
                else:
                    abso_material = 'GypsumPlasterBoard'
            elif room == 'Hallway':
                if wall == 'Floor':
                    abso_material = 'ConcreteFloor'
                else:
                    abso_material = 'Brick'
            elif room == 'Meeting_room':
                if wall == 'Floor':
                    abso_material = 'WoodFloor'
                elif wall == 'Ceiling':
                    abso_material = 'WoodenTongueGrooveCeiling'
                else:
                    abso_material = 'WoodSidewalls20'
            elif room == 'Cinema':
                if wall == 'Floor':
                    abso_material = 'CarpetHeavyOnConcrete'
                else:
                    abso_material = 'CurtainsHeavy'
            elif room == 'Restroom':
                abso_material = 'Marble'

            if room == 'Outdoors':
                scat_material = 'Generic_rough'
            elif room == 'Large_exhibit':
                scat_material = 'Smooth_painted_concrete'
            elif room == 'Small_exhibit':
                scat_material = 'Brickwork_filled'
            elif room == 'Ballroom':
                if wall == 'Floor':
                    scat_material = 'Generic_smooth'
                else:
                    scat_material = 'Generic_rough'
            elif room == 'Hallway':
                scat_material = 'Brickwork_open'
            elif room == 'Meeting_room':
                scat_material = 'Bookshelf'
            elif room == 'Cinema':
                scat_material = 'Generic_rough'
            elif room == 'Restroom':
                scat_material = 'Smooth_painted_concrete'

            abso_material_coeffs = reference_absorptions[abso_material]
            scat_material_coeffs = reference_scatterings[scat_material]

            # Add two octave bands (8k and 16k).
            file.write(raves_material + ', ' + ', '.join(abso_material_coeffs) + ', ' + abso_material_coeffs[-1] + ', ' + abso_material_coeffs[-1] + '\n')
            # Use the maximum recommended value.
            file.write(raves_material + ', ' + scat_material_coeffs[-1] + '\n')
