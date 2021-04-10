
def detect_vehicles(img, model, scaler):

    import params as p
    import helpers as h
    import cv2 as cv
    import numpy as np
    from heatmap import Heatmap

    # Perform HOG extraction
    print("Extracting HOG features...")
    hog_image = h.hog_extraction(img, p.hog_params)

    window_dims = []
    windows = []

    print("Generating sliding window features...")

    for i in range(len(p.slider_params['sizes'])):
        x = 0
        y = p.slider_params['positions'][i]
        size = p.slider_params['sizes'][i]
        
        while x + size < img.shape[1]:
            window_dims.append([y, x, size])
            window = hog_image[y:y+size, x:x+size]
            window = cv.resize(window, (80, 80))
            window = window.flatten()
            
            windows.append(window)
            x += p.slider_params['step']
    window_dims = np.asarray(window_dims)

    # Normalize windows into features
    windows = np.asarray(windows)
    features = scaler.transform(windows)

    # Feed into SVM model and store positive labels
    print("Identifying positive features...")
    results = model.predict(features)

    labels = []
    for i in range(len(results)):
        if int(results[i]) == 1:
            labels.append(i)
            
    labels = np.asarray(labels)

    # Windows that were all labelled as positive from SVM
    pw = window_dims[labels]

    # Get heatmap using the labels
    hm = Heatmap(img.shape[0], img.shape[1])
    hm.generate(pw)
    heatmap = hm.get().astype('uint8')

    # Perform thresholding on heatmap
    ret, detected = cv.threshold(heatmap,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Extract rectangles from threshold
    print('Extracting detected vehicles...')
    contours = cv.findContours(detected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    bboxes = []
    minSize = 80*80
    maxSize = 200*200

    # Format the rectangles into the bbox label format
    for item in contours:
        x,y,w,h = cv.boundingRect(item)
        if w*h > minSize and w*h < maxSize:
            bbox = {'bbox': {'top':y,
                             'left':x,
                             'bottom': y+h,
                             'right': x+w}}
            bboxes.append(bbox)

    print('Detection completed')
    return bboxes