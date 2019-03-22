import cv2
import dlib
from imutils import face_utils

# face detector
face_detect = dlib.get_frontal_face_detector()
# facial landmarks detector
face_marks = dlib.shape_predictor("./utils/shape_predictor_68_face_landmarks.dat")

def get_face(gray):
    """
    Arguments:
        gray: grayscale image array
    Returns:
        rectangle coordinates for one face if found, else None
    """
    face_coords = face_detect(gray, 0)
    if len(face_coords)==0:
        return
    return face_coords[0]

def get_marks(gray, face_coord):
    """
    Arguments:
        gray: grayscale image array
        face_coord: rectangle coordinates for one face
    Returns:
        array of coordinates of facial landmarks scaled to range [-0.5, 0.5]
    """
    shape = face_marks(gray, face_coord)
    shape = face_utils.shape_to_np(shape).astype(float)
    shape[:,0] = (shape[:,0] - face_coord.left())/face_coord.width()-0.5
    shape[:,1] = (shape[:,1] - face_coord.top())/face_coord.height()-0.5
    return shape

if __name__=='__main__':
    img = cv2.imread("img.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coord = get_face(gray)
    marks = get_marks(gray, face_coord)
    print(marks)
