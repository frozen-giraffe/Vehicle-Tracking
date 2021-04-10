from skimage.feature import hog

# Perform HOG feature extraction
def hog_extraction(img, hp):
    features, hog_image = hog(  img,
                                orientations=hp['orientations'],
                                pixels_per_cell=hp['pixels_per_cell'],
                                cells_per_block=hp['cells_per_block'],
                                transform_sqrt=hp['transform_sqrt'],
                                visualize=hp['visualize'],
                                multichannel=hp['multichannel'])
    return hog_image