from models import FaceRecognition

def main():
    print("Homework #8")
    print("[Face recognition using PCA]")
    exp = FaceRecognition(
        face_root="data/att_faces",
        target_data = "myface.png",
        pre_extract_pca_dict = "pca_data_dict.pkl",
    )
    exp()

if __name__=="__main__":
    main()

