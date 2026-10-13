import math
import sys

import cv2
import numpy as np

import Preprocess

# Tham số cho bước nhị phân hóa thích nghi (Adaptive Threshold)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1  # Biến đếm số thứ tự biển số tìm thấy trong ảnh

# Ngưỡng tỷ lệ diện tích của ký tự so với diện tích biển số
Min_char = 0.01
Max_char = 0.09

# Kích thước chuẩn hóa của ảnh ký tự đưa vào mô hình KNN
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Đọc ảnh đầu vào và chuẩn hóa kích thước về Full HD (1920x1080)
img_path = "data/image/10.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"Lỗi: Không tìm thấy hoặc không đọc được file ảnh tại '{img_path}'")
    sys.exit(1)

img = cv2.resize(img, dsize=(1920, 1080))

######## Tải dữ liệu và huấn luyện mô hình KNN ######################
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # Đưa mảng nhãn về 1 chiều để truyền vào hàm train
kNearest = cv2.ml.KNearest_create()  # Khởi tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
#####################################################################

################ Tiền xử lý ảnh #################
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Phát hiện biên bằng thuật toán Canny
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Phép giãn nở (Dilation) làm liền nét biên
# cv2.imshow("dilated_image", dilated_image)

###### Tìm đường viền (contour) và lọc vùng biển số xe #############
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contour có diện tích lớn nhất
# cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các contour trong hình lớn

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi của contour
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Xấp xỉ đa giác, chỉ giữ lại các contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    # cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    # cv2.putText(img, str(ratio), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    if (len(approx) == 4):
        screenCnt.append(approx)
        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if len(screenCnt) == 0:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for cnt_plate in screenCnt:
        cv2.drawContours(img, [cnt_plate], -1, (0, 255, 0), 3)  # Khoanh vùng khung biển số xe màu xanh lá

        ############## Tính góc nghiêng của biển số #####################
        (x1, y1) = cnt_plate[0, 0]
        (x2, y2) = cnt_plate[1, 0]
        (x3, y3) = cnt_plate[2, 0]
        (x4, y4) = cnt_plate[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        array.sort(reverse=True, key=lambda x: x[1])  # Sắp xếp tọa độ theo trục y giảm dần
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        if ke == 0:
            angle = 90.0
        else:
            angle = math.atan(doi / ke) * (180.0 / math.pi)
        #################################################################

        ########## Cắt vùng biển số và xoay thẳng góc ###################
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [cnt_plate], 0, 255, -1)
        # cv2.imshow("new_image", new_image)

        # Xác định tọa độ giới hạn và cắt (crop) vùng biển số
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        # Tạo ma trận xoay để căn chỉnh biển số nằm ngang
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

        # Phóng to vùng biển số gấp 3 lần để hỗ trợ tách ký tự rõ nét hơn
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        #################### Tiền xử lý vùng biển số và phân tách ký tự ####################
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ đường viền các ký tự bên trong biển số

        ##################### Lọc ra các contour là ký tự hợp lệ #################
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h
            # cv2.putText(roi, str(char_area), (x, y+20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            # cv2.putText(roi, str(ratiochar), (x, y+20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

            # Kiểm tra điều kiện diện tích và tỷ lệ rộng/cao (0.25 - 0.7) của ký tự
            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Tăng nhẹ x nếu trùng tọa độ để vẫn lưu và vẽ được đầy đủ các ký tự
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

                # cv2.putText(roi, str(char_area), (x, y+20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        ############ Nhận dạng từng ký tự bằng KNN ##########################
        char_x = sorted(char_x)  # Sắp xếp ký tự theo thứ tự từ trái sang phải
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]  # Cắt rời ảnh của ký tự

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # Chuẩn hóa kích thước về 20x30
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # Làm phẳng ảnh thành vector 1 chiều (1x600)

            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)  # Gọi hàm dự đoán KNN với k = 3
            strCurrentChar = str(chr(int(npaResults[0][0])))  # Chuyển mã ASCII sang ký tự tương ứng
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            # Phân chia ký tự thuộc dòng 1 hay dòng 2 của biển số dựa trên vị trí trục y
            if (y < height / 3):
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # cv2.putText(img, first_line + "-" + second_line, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1

# Thu nhỏ ảnh gốc 50% để hiển thị vừa màn hình và chờ nhấn phím bất kỳ để thoát
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img)

cv2.waitKey(0)
cv2.destroyAllWindows()