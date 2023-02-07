import numpy as np
import cv2 as cv

class LSDExtractor:
    def __init__(self, scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, log_eps=0, density_th=0.7, n_bins=1024,
                 runtime_scale=2, runtime_num_octave=1):

        params = cv.line_descriptor.LSDParam()
        params.scale       = scale
        params.sigma_scale = sigma_scale
        params.quant       = quant
        params.ang_th      = ang_th
        params.log_eps     = log_eps
        params.density_th  = density_th
        params.n_bins      = n_bins

        self.extractor = cv.line_descriptor.LSDDetector.createLSDDetectorWithParams(params)
        self.runtime_scale = runtime_scale
        self.runtime_num_octave = runtime_num_octave

    def extract(self, image, scale=None, num_octaves=None, mask=None):
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if scale is None:
            scale = self.runtime_scale
        if num_octaves is None:
            num_octaves = self.runtime_num_octave
        keylines = self.extractor.detect(image, scale, num_octaves, mask)
        lines = np.array([[l.startPointX, l.startPointY, l.endPointX, l.endPointY] for l in keylines])
        return lines

class FLDExtractor:
    def __init__(self, length_threshold=10, distance_threshold=1.414213562, canny_th1=50.0, canny_th2=50.0, canny_aperture_size=3, do_merge=False):
        self.extractor = cv.ximgproc.createFastLineDetector(length_threshold=length_threshold, distance_threshold=distance_threshold,
                                                            canny_th1=canny_th1, canny_th2=canny_th2, canny_aperture_size=canny_aperture_size, do_merge=do_merge)

    def extract(self, image):
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        lines = self.extractor.detect(image)
        return lines.reshape(-1, 4)

def draw_line_segments(image, lines, color=(0, 0, 255), thickness=1):
    pair_list = lines.astype(np.int32)
    for pair in pair_list:
        cv.line(image, pair[0:2], pair[2:4], color=color, thickness=thickness)



if __name__ == '__main__':
    # Test line segment extractors
    image = cv.imread('data/220720_M327/calib.png')

    extractors = [
        ('LSD', LSDExtractor()),
        ('FLD', FLDExtractor()),
    ]
    for name, extractor in extractors:
        img = image.copy()
        lines = extractor.extract(img)
        draw_line_segments(img, lines)
        cv.imshow(name, img)

    cv.waitKey(0)
    cv.destroyAllWindows()
