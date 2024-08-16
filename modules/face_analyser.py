from typing import Any, Optional
import insightface
import modules.globals
from modules.typing import Frame

# Global variable to hold the face analyser instance
FACE_ANALYSER = None

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        # Initialize the FaceAnalysis object with the specified model
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        
        # Prepare the analyser with the desired context (e.g., GPU/CPU) and detection size
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    
    return FACE_ANALYSER

def get_one_face(frame: Frame) -> Optional[Any]:
    # Get faces detected in the frame
    faces = get_face_analyser().get(frame)
    
    try:
        # Return the face with the smallest x-coordinate (i.e., the leftmost face)
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        # Return None if no face is detected
        return None

def get_many_faces(frame: Frame) -> Optional[Any]:
    try:
        # Return all detected faces in the frame
        return get_face_analyser().get(frame)
    except IndexError:
        # Return None if no faces are detected
        return None
