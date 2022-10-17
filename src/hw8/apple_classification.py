from models import AppleClassification

def main():
    print("Homework #8")
    print("[Apple Classification using PCA]")
    exp = AppleClassification(
        data_a = "data/data_a.txt",
        data_b = "data/data_b.txt",
        test = "data/test.txt"
    )
    exp()

if __name__=="__main__":
    main()

