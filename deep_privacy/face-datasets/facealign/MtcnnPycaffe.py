from __future__ import print_function

import os,sys
os.environ['GLOG_minloglevel'] = '2'  # Hide caffe debug info.
sys.path.append('/usr/lib/python3/dist-packages/caffe/')
import math
import time

import cv2
import caffe

import numpy as np
import sys

from alignment import alignface_96x112

def draw_and_show(im, bboxes, points=None):
    '''Draw bboxes and points on image, and show.

    Args:
      im: image to draw on.
      bboxes: (tensor) bouding boxes sized [N,4].
      points: (tensor) landmark points sized [N,10],
        coordinates arranged as [x,x,x,x,x,y,y,y,y,y].
    '''
    print('Drawing..')

    num_boxes = len(bboxes)
    for i in range(num_boxes):
        box = bboxes[i]
        if box[0] != None:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            print('Rect:', x1, y1,  x2,y2)
            # As im is rotated, so need to swap x and y.
            cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,255), 2)

        if len(points):
            p = points[i]
            for i in range(5):
                if p[i] != None:
                    x = int(p[i])
                    y = int(p[i+5])
                    # Again, swap x and y.
                    cv2.circle(im, (x,y), 1, (0,0,255), 2)

    cv2.imshow('result', im)
    #cv2.waitKey(0)

def non_max_suppression(bboxes, threshold=0.5, mode='union'):
    '''Non max suppression.

    Args:
      bboxes: (tensor) bounding boxes and scores sized [N, 5].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      Bboxes after nms.
      Picked indices.

    Ref:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode )

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return bboxes[keep], keep

def padding(bboxes, im_height, im_width):
    '''Padding bouding boxes the edge of image, if it's too large.'''
    bboxes[:,0] = np.maximum(0, bboxes[:,0])
    bboxes[:,1] = np.maximum(0, bboxes[:,1])
    bboxes[:,2] = np.minimum(im_width-1, bboxes[:,2])
    bboxes[:,3] = np.minimum(im_height-1, bboxes[:,3])
    return bboxes

def bbox_to_square(bboxes):
    '''Make bounding boxes square.'''
    square_bbox = bboxes.copy()

    w = bboxes[:,2] - bboxes[:,0] + 1
    h = bboxes[:,3] - bboxes[:,1] + 1
    max_side = np.maximum(h,w)

    square_bbox[:,0] = bboxes[:,0] + (w - max_side) * 0.5
    square_bbox[:,1] = bboxes[:,1] + (h - max_side) * 0.5
    square_bbox[:,2] = square_bbox[:,0] + max_side - 1
    square_bbox[:,3] = square_bbox[:,1] + max_side - 1

    return square_bbox

def bbox_regression(bboxes):
    '''Bounding box regression.

    Args:
      bboxes: (tensor) bounding boxes sized [N,9], containing:
        x1, y1, x2, y2, score, regy1, regx1, regy2, regx2.

    Return:
      Regressed bounding boxes sized [N,5].
    '''
    bbw = bboxes[:,2] - bboxes[:,0] + 1
    bbh = bboxes[:,3] - bboxes[:,1] + 1

    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    scores = bboxes[:,4]

    # Note the sequence.
    rgy1 = bboxes[:,5]
    rgx1 = bboxes[:,6]
    rgy2 = bboxes[:,7]
    rgx2 = bboxes[:,8]

    ret = np.vstack([x1 + rgx1 * bbw,
                     y1 + rgy1 * bbh,
                     x2 + rgx2 * bbw,
                     y2 + rgy2 * bbh,
                     scores])
    return ret.T

def get_pnet_boxes(outputs, scale, threshold):
    '''Generate bouding boxes from PNet outputs.

    Args:
      outputs: (dict) PNet outputs.
      scale: (float) image scale ration.
      threshold: (float) confidence threshold.

    Returns:
      A tensor representing generated bounding boxes sized [N,9].
    '''
    confidence = outputs['prob1'][0][1]  # [H,W]
    regression = outputs['conv4-2'][0]   # [4,H,W]

    # Filter out confidence > threshold.
    # Note:
    #   y is the row-index.
    #   x is the col-index.
    y, x = np.where(confidence > threshold)

    # Get regression outputs.
    reg_y1, reg_x1, reg_y2, reg_x2 = [regression[i,y,x] for i in range(4)]
    reg = np.array([reg_x1, reg_y1, reg_x2, reg_y2])

    # Get scores.
    scores = confidence[y,x]  # [N,]

    # Get face rects.
    stride = 2
    cell_size = 12

    x1 = np.round((stride*x+1) / scale)
    y1 = np.round((stride*y+1) / scale)
    x2 = np.round((stride*x+1 + cell_size-1) / scale)
    y2 = np.round((stride*y+1 + cell_size-1) / scale)
    rect = np.array([x1, y1, x2, y2])

    bbox = np.vstack([rect, scores, reg])  # [9,N]
    return bbox.T  # [N,9]

def get_rnet_boxes(bboxes, outputs, threshold):
    '''Generate bounding boxes from RNet outputs.

    Args:
      bboxes: (tensor) PNet bouding boxes sized [N,5].
      outputs: (dict) RNet outputs.
      threshold: (float) confidence threshold.

    Returns:
      A tensor representing generated bounding boxes sized [N,9].
    '''
    confidence = outputs['prob1'][:,1]
    regression = outputs['conv5-2']

    indices = np.where(confidence > threshold)
    rects = bboxes[indices][:,0:4]  # [N,4]
    scores = confidence[indices]    # [N,]
    scores = scores.reshape(-1,1)   # [N,1]
    regs = regression[indices]      # [N,4]

    return np.hstack([rects, scores, regs])  # [N,9]

def get_onet_boxes(bboxes, outputs, threshold):
    '''Generate bounding boxes and points from ONet outputs.

    Args:
      bboxes: (tensor) RNet bounding boxes sized [N,5].
      outputs: (dict) ONet outputs.
      threshold: (float) confidence threshold.

    Returns:
      A tensor representing generated bounding boxes sized [N,9].
      A tensor representing points sized [N,10].
    '''
    confidence = outputs['prob1'][:,1]
    regression = outputs['conv6-2']
    points = outputs['conv6-3']

    indices = np.where(confidence > threshold)

    rects = bboxes[indices][:,0:4]
    scores = confidence[indices]
    scores = scores.reshape(-1,1)
    regs = regression[indices]

    points = points[indices]   # Note `y` is in the front.
    points_y = points[:,0:5]   # [N,5]
    points_x = points[:,5:10]  # [N,5]

    w = rects[:,2] - rects[:,0] + 1
    h = rects[:,3] - rects[:,1] + 1

    x1 = rects[:,0]
    y1 = rects[:,1]

    points_x = points_x * w.reshape(-1,1) + x1.reshape(-1,1)
    points_y = points_y * h.reshape(-1,1) + y1.reshape(-1,1)

    # We move `x` ahead, points=[x,x,x,x,x,y,y,y,y,y].
    return np.hstack([rects, scores, regs]), np.hstack([points_x, points_y])

def get_inputs_from_bboxes(im, bboxes, size):
    '''Get network inputs based on generated bounding boxes.

    Args:
      im: (image) rotated original image.
      bboxes: (tensor) regressed bounding boxes sized [N,5].
      size: (int) expected input size.

    Returns:
      A tensor sized [N, 3, size, size].
    '''
    num_boxes = bboxes.shape[0]
    inputs = np.zeros((num_boxes, size, size, 3), dtype=np.float32)
    for i in range(num_boxes):
        x1 = int(bboxes[i,0])
        y1 = int(bboxes[i,1])
        x2 = int(bboxes[i,2])
        y2 = int(bboxes[i,3])

        im_crop = im[y1:y2+1, x1:x2+1, :]
        inputs[i] = cv2.resize(im_crop, (size,size))

    # [N,H,W,C] -> [N,C,H,W] to meet the Caffe needs.
    inputs = np.transpose(inputs, (0,3,1,2))

    # Zero mean and normalization.
    inputs = (inputs - 127.5) * 0.0078125
    return inputs


class MtcnnDetector(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a PyCaffe version
    """
    def __init__(self,
                 model_folder='./model/',
                 minsize = 20,
                 threshold = [0.6, 0.7, 0.7],
                 factor = 0.709,
                 gpu_id = 0):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : string
                    path for the models
                minsize : float number
                    minimal face to detect
                threshold : float number
                    detect threshold for 3 stages
                factor: float number
                    scale factor for image pyramid
                num_worker: int number
                    number of processes we use for first stage
                accurate_landmark: bool
                    use accurate landmark localization or not

        """
        caffe.set_mode_cpu()
        #caffe.set_device(gpu_id)
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_folder = cwd + '/model/'
        # Load models.
        prototxt = [model_folder + x + '.prototxt' for x in ['det1', 'det2', 'det3']]
        binary = [model_folder + x + '.caffemodel' for x in ['det1', 'det2', 'det3']]
        self.PNet = caffe.Net(prototxt[0], binary[0], caffe.TEST)
        self.RNet = caffe.Net(prototxt[1], binary[1], caffe.TEST)
        self.ONet = caffe.Net(prototxt[2], binary[2], caffe.TEST)

        self.minsize   = float(minsize)
        self.factor    = float(factor)
        self.threshold = threshold
        
        
    def detect_face(self, im):
        """
            detect face over img
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
        Retures:
        -------
            bboxes: numpy array, n x 5 (x1,y1,x2,y2,score)
                bboxes
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                landmarks
        """
        im = im.astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (1,0,2))  # Rotate image.

        image_height, image_width, num_channels = im.shape
        #print('Image shape:', im.shape)
        # convert gray to rgb
        if num_channels != 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
 
        MIN_FACE_SIZE = self.minsize    # Minimum face size.
        MIN_INPUT_SIZE = 12.   # Minimum input size.
        MAX_INPUT_SIZE = 1024
        m = MIN_INPUT_SIZE / MIN_FACE_SIZE

        min_size = min(image_height, image_width)
        min_size = min_size * m

        scales = []
        counter = 0
        FACTOR = self.factor
        while min_size >= MIN_INPUT_SIZE:
            if min_size <= MAX_INPUT_SIZE:
                scales.append(m * FACTOR**counter)
            min_size = min_size * FACTOR
            counter = counter + 1
            
        # Threshold for each stage.
        THRESHOLD = self.threshold
        PNet = self.PNet
        RNet = self.RNet
        ONet = self.ONet
        t1 = time.time()

        # --------------------------------------------------------------
        # First stage.
        #
        total_boxes = []  # Bounding boxes of all scales.
        for scale in scales:
            hs = int(math.ceil(image_height*scale))
            ws = int(math.ceil(image_width*scale))

            im_resized = cv2.resize(im, (ws,hs), interpolation=cv2.INTER_AREA)
            #print('Resize to:', im_resized.shape)

            # H,W,C -> C,H,W
            im_resized = np.transpose(im_resized, (2,0,1))

            # Zero mean and normalization.
            im_resized = (im_resized - 127.5) * 0.0078125

            # Reshape input layer.
            PNet.blobs['data'].reshape(1, 3, hs, ws)
            PNet.blobs['data'].data[...] = im_resized
            outputs = PNet.forward()

            bboxes = get_pnet_boxes(outputs, scale, THRESHOLD[0])
            bboxes,_ = non_max_suppression(bboxes, 0.5)

            total_boxes.append(bboxes)

        total_boxes = np.vstack(total_boxes)

        bboxes,_ = non_max_suppression(total_boxes, 0.7)
        bboxes = bbox_regression(total_boxes)

        bboxes = bbox_to_square(bboxes)
        bboxes = padding(bboxes, image_height, image_width)

        #print('After PNet bboxes shape: ', bboxes.shape)
        if bboxes.shape[0] == 0:
            return [],[]

        # --------------------------------------------------------------
        # Second stage.
        #
        inputs = get_inputs_from_bboxes(im, bboxes, 24)
        N,C,H,W = inputs.shape

        RNet.blobs['data'].reshape(N,3,H,W)
        RNet.blobs['data'].data[...] = inputs
        outputs = RNet.forward()

        bboxes = get_rnet_boxes(bboxes, outputs, THRESHOLD[1])

        bboxes,_ = non_max_suppression(bboxes, 0.7)
        bboxes = bbox_regression(bboxes)
        bboxes = bbox_to_square(bboxes)
        bboxes = padding(bboxes, image_height, image_width)

        #print('After RNet bboxes shape: ', bboxes.shape)
        if bboxes.shape[0] == 0:
            return [],[]

        # --------------------------------------------------------------
        # Third stage.
        #
        inputs = get_inputs_from_bboxes(im, bboxes, 48)
        N,C,H,W = inputs.shape

        ONet.blobs['data'].reshape(N,3,H,W)
        ONet.blobs['data'].data[...] = inputs
        outputs = ONet.forward()

        bboxes, points = get_onet_boxes(bboxes, outputs, THRESHOLD[2])
        bboxes = bbox_regression(bboxes)

        bboxes, picked_indices = non_max_suppression(bboxes, 0.7, 'min')
        points = points[picked_indices]
        bboxes = padding(bboxes, image_height, image_width)

        #print('After ONet bboxes shape: ', bboxes.shape, '\n')
        if bboxes.shape[0] == 0:
            return [],[]

        t2 = time.time()
        #print('Total time: %.3fs\n' % (t2-t1))
        # transpose points
        for i in range(len(points)):
            for j in range(5):
                t = points[i][j]
                points[i][j] = points[i][j+5]
                points[i][j+5] = t
        # tranpose bboxes  
        for i in range(len(bboxes)):
            t = bboxes[i][0]
            bboxes[i][0] = bboxes[i][1]
            bboxes[i][1] = t
            t = bboxes[i][2]
            bboxes[i][2] = bboxes[i][3]
            bboxes[i][3] = t 
            
        return bboxes, points

        
if __name__ == '__main__':
    detector = MtcnnDetector()
    # Load image.
    im = cv2.imread(sys.argv[1])
    bboxes,points = detector.detect_face(im)
    #draw_and_show(im, bboxes, points)
    aligned = alignface_96x112(im, points)
    cv2.imwrite('align.png', aligned[0])
