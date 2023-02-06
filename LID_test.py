import cv2 as cv

class LID():
    
    def lsd(self, img, scale=2, octaves=1):
        ls_d = cv.line_descriptor.LSDDetector.createLSDDetector()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lines = ls_d.detect(gray, scale, octaves)
        result = result = cv.line_descriptor.drawKeylines(img, lines, color=(0, 0, 255))
        return result
    
    def fsd(self, img, len_th=15, c_th1=20, c_th2=60):
        fs_d = cv.ximgproc.createFastLineDetector(length_threshold=len_th, canny_th1=c_th1, canny_th2=c_th2)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lines = fs_d.detect(gray)
        result = fs_d.drawSegments(img.copy(), lines, linecolor=(0, 0, 255))
        return result
    
if __name__ == '__main__':
    img = cv.imread('rail.jpg')
    
    lid = LID()
    result = lid.fsd(img)
    
    cv.imshow('img', result)
    key = cv.waitKey(0)
    cv.destroyAllWindows()