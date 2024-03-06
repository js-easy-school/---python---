import cv2
import numpy as np

original = cv2.imread("1.jpg")
image_to_compare = cv2.imread("images/1_result2.jpg")



# 1) Проверка, равны ли 2 изображения
if original.shape == image_to_compare.shape:
    print("Изображения имеют одинаковый размер и количество каналов")
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("Изображения полностью идентичны")
    else:
        print("Изображения не идентичны")

# 2) Проверка на схожесть между 2 изображениями
sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

print("Keypoints 1ST Image: " + str(len(kp_1)))
print("Keypoints 2ND Image: " + str(len(kp_2)))

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)



print("GOOD Matches:", len(good_points))

print("How good it's the match: ", len(good_points) / number_keypoints * 100)

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))

cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
cv2.imshow("Duplicate", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
cv2.waitKey(0)
cv2.destroyAllWindows()
