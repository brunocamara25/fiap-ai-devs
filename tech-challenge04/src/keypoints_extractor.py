def extract_keypoints(landmarks, key='nose_tip'):
    """
    Extracts a specific keypoint from face landmarks.
    If the key is not present, returns (0, 0).
    """
    return landmarks[key][0] if key in landmarks else (0, 0)