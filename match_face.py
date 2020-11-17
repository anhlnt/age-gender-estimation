from imgcmp import FuzzyImageCompare
from PIL import Image
import numpy as np

def match_point(img1, img2):
    cmp = FuzzyImageCompare(Image.fromarray(img1), Image.fromarray(img2))
    sim = cmp.similarity()
    return sim

def match(face, faces, faces_cur, id_cur, age, gender):
    idx = id_cur
    if len(faces) > 0:
        matchPoint = []
        for i in range(len(faces)):
            matchPoint.append(match_point(faces[i][0], face))
        argmax = np.argmax(matchPoint)
        print(matchPoint[argmax])
        if matchPoint[argmax] > 45:
            idx = faces[argmax][1]
            faces_cur.append((face, idx, age, gender))
            faces.pop(argmax)
        else:
            print("matchPoint　が低すぎ", matchPoint[argmax])
            faces_cur.append((face, idx, age, gender))
            return -1
    else:
        faces_cur.append((face, idx, age, gender))
        return -1
    return idx


def test(a, b):
    a.append(1)
    b.pop(0)
    return a, b

def main():
    a = [[0]]
    b = [-1]
    a, b = test(a, b)
    print(a, b)




if __name__ == "__main__":
    main()