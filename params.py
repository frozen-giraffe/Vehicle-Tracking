    

# HOG Parameters
hog_params =    {  
                'orientations': 12,
                'pixels_per_cell': (12, 12),
                'cells_per_block': (2, 2),
                'transform_sqrt': True, #?
                'visualize': True, #?
                'multichannel': True #?
                }

# Parameters for getting negative samples
subimage_params =   { 
                    'subimage-size': 60,
                    'horizon': 320,
                    }

# Sliding window parameters
slider_params= {'sizes': [40, 40, 60, 80, 120],
               'step': 10,
               'positions': [330, 340, 330, 340, 340],
               'colours': [(255, 0, 0),
                           (0, 255, 0),
                           (0, 0, 255),
                           (255, 255, 0),
                           (0, 255, 255)]}