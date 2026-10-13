# 🚗 Nhận dạng biển số xe bằng OpenCV và KNN

Dự án phát hiện và nhận dạng biển số xe (số và chữ cái) từ **ảnh tĩnh** và **video**, sử dụng các kỹ thuật xử lý ảnh của OpenCV kết hợp thuật toán phân loại **K-Nearest Neighbors (KNN)**.

---

## 📑 Mục lục

1. [Tính năng](#-tính-năng)
2. [Quy trình xử lý](#-quy-trình-xử-lý)
3. [Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
5. [Cài đặt](#-cài-đặt)
6. [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
7. [Huấn luyện lại dữ liệu KNN](#-huấn-luyện-lại-dữ-liệu-knn)
8. [Tham số có thể điều chỉnh](#-tham-số-có-thể-điều-chỉnh)
9. [Đánh giá kết quả](#-đánh-giá-kết-quả)
10. [Hạn chế và hướng cải thiện](#-hạn-chế-và-hướng-cải-thiện)
11. [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## ✨ Tính năng

- Nhận dạng biển số từ **ảnh** và hiển thị kết quả trực quan (khung biển số, từng ký tự, chuỗi nhận dạng).
- Nhận dạng biển số từ **video** theo thời gian thực, đếm số biển số tìm thấy và tính tỷ lệ.
- Hỗ trợ biển số **một dòng** và **hai dòng** (tự phân chia dòng theo vị trí ký tự).
- Tự động **xoay thẳng** biển số bị nghiêng trước khi tách ký tự.
- Ghi video kết quả ra file `output_1920x1080.mp4` (độ phân giải 1920x1080).
- Bộ dữ liệu KNN gồm **32 lớp ký tự** (160 mẫu), có thể tự tạo lại bằng `GenData.py`.

---

## 🔄 Quy trình xử lý

```
Ảnh/Khung hình  →  Tiền xử lý  →  Tìm biển số  →  Cắt & xoay  →  Tách ký tự  →  KNN  →  Kết quả
```

| Bước | Mô tả | Kỹ thuật |
|------|-------|----------|
| 1. Tiền xử lý | Chuyển ảnh sang kênh **Value** (HSV), tăng tương phản bằng Top-hat và Black-hat, làm mờ Gauss 5x5, nhị phân hóa thích nghi | `Preprocess.py` |
| 2. Phát hiện biển số | Canny, giãn nở (dilate), tìm contour, lấy 10 contour lớn nhất, xấp xỉ đa giác và giữ contour có **4 cạnh** | `findContours`, `approxPolyDP` |
| 3. Lọc theo tỷ lệ (video) | Giữ contour có tỷ lệ w/h thuộc `0.8–1.5` (biển vuông) hoặc `4.5–6.5` (biển dài) | `Video_test2.py` |
| 4. Cắt và xoay | Tạo mask, cắt vùng biển số, tính góc nghiêng và xoay thẳng, phóng to 3 lần | `warpAffine` |
| 5. Tách ký tự | Giãn nở, tìm contour ngoài, lọc theo diện tích và tỷ lệ w/h (`0.25–0.7`) | `findContours` |
| 6. Nhận dạng | Resize mỗi ký tự về **20x30**, làm phẳng thành vector 600 chiều, phân loại bằng KNN (`k=3`) | `cv2.ml.KNearest` |
| 7. Ghép kết quả | Ký tự có `y < height/3` thuộc dòng 1, còn lại thuộc dòng 2; sắp xếp theo trục x | — |

---

## 📁 Cấu trúc dự án

```
.
├── GenData.py              # Tạo dữ liệu huấn luyện KNN từ training_chars.png
├── Preprocess.py           # Các hàm tiền xử lý ảnh (extractValue, maximizeContrast, preprocess)
├── Image_test2.py          # Nhận dạng biển số từ ảnh
├── Video_test2.py          # Nhận dạng biển số từ video
├── classifications.txt     # Nhãn (mã ASCII) của các mẫu huấn luyện
├── flattened_images.txt    # Dữ liệu ảnh ký tự đã làm phẳng (mỗi dòng 600 giá trị)
├── training_chars.png      # Ảnh chứa các ký tự mẫu (đầu vào của GenData.py)
├── setup.txt               # Danh sách thư viện cần cài
├── measurementCriteria.txt # Tiêu chí đo lường kết quả
├── README.md               # Tài liệu này
└── data/
    ├── image/              # Ảnh kiểm thử (ví dụ: 10.jpg)
    ├── video/              # Video kiểm thử (ví dụ: video1.mp4)
    └── README-VIDEO.txt    # Hướng dẫn nếu không phát được video
```

---

## 💻 Yêu cầu hệ thống

- **Python** >= 3.8 (Tương thích tốt với cả Python 3.12, 3.13)
- Các thư viện (theo `setup.txt`):

| Thư viện | Phiên bản | Công dụng |
|----------|-----------|-----------|
| `opencv-python` | `== 4.11.0.86` | Phát hiện và nhận dạng biển số (`cv2.ml`) |
| `numpy` | `>= 1.22.4` | Xử lý dữ liệu ảnh dạng mảng |
| `imutils` | `>= 0.5.4` | Hỗ trợ xử lý ảnh |
| `matplotlib` | `>= 3.5.1` | Hiển thị và đánh giá |
| `scikit-learn` | `>= 1.0.2` | Hỗ trợ KNN / đánh giá |

> **Lưu ý:**
> - Mã nguồn hiện tại chỉ import trực tiếp `cv2`, `numpy` và `math`. Các thư viện còn lại dùng cho mở rộng và đánh giá.
> - Bắt buộc cài đặt `opencv-python==4.11.0.86` (không cài bản `5.x` trở lên vì module `cv2.ml` đã bị lược bỏ khỏi gói mặc định).

---

## ⚙️ Cài đặt

```bash
# 1. Tải và giải nén dự án (hoặc clone)
unzip project.zip && cd project

# 2. (Khuyến nghị) Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate           # Windows (CMD / PowerShell)
source venv/Scripts/activate    # Windows (Git Bash / MINGW64)
source venv/bin/activate        # Linux / macOS

# 3. Cài đặt thư viện
pip install -r setup.txt
# Hoặc chạy trực tiếp:
pip install opencv-python==4.11.0.86 numpy imutils matplotlib scikit-learn
```

Trước khi chạy, hãy kiểm tra:

- Các file `classifications.txt`, `flattened_images.txt` nằm cùng thư mục với file `.py`.
- Đường dẫn ảnh (`data/image/10.jpg`) và video (`data/video/video1.mp4`) trong code là chính xác.

---

## ▶️ Hướng dẫn sử dụng

### Nhận dạng từ ảnh

```bash
python Image_test2.py
```

- Ảnh được resize về 1920x1080.
- Hiển thị cửa sổ ảnh gốc có khoanh vùng biển số, cùng cửa sổ riêng cho từng biển số với các ký tự đã nhận dạng.
- Kết quả in ra terminal, ví dụ: `License Plate 1 is: 59A1 - 12345`.
- Nhấn phím bất kỳ để đóng các cửa sổ.

Để đổi ảnh kiểm thử, sửa dòng:

```python
img_path = "data/image/10.jpg"
```

### Nhận dạng từ video

```bash
python Video_test2.py
```

- Video được đọc từ `data/video/video1.mp4` (đổi đường dẫn ở dòng `cv2.VideoCapture(...)`).
- Mỗi khung hình được xử lý; chỉ những biển số có **7–9 ký tự** hợp lệ mới được chấp nhận và đếm.
- Terminal hiển thị: số biển số tìm thấy, tổng số khung hình, tỷ lệ tìm thấy.
- Video kết quả được lưu tại `output_1920x1080.mp4`.
- Nhấn phím **`q`** để thoát.

> Nếu không phát được video, xem `data/README-VIDEO.txt`.

---

## 🧠 Huấn luyện lại dữ liệu KNN

Nếu muốn thay đổi bộ ký tự mẫu:

1. Chuẩn bị ảnh `training_chars.png` chứa các ký tự mẫu (nền sáng, chữ tối).
2. Chạy:
   ```bash
   python GenData.py
   ```
3. Với mỗi ký tự được khoanh khung đỏ, **nhấn phím tương ứng** trên bàn phím (`0–9`, `A–Z` viết hoa).
   - Nhấn `ESC` để thoát.
   - Phím không hợp lệ sẽ bỏ qua mẫu đó.
4. Chương trình ghi ra hai file:
   - `classifications.txt`: nhãn (mã ASCII) của từng mẫu.
   - `flattened_images.txt`: ảnh ký tự 20x30 đã làm phẳng.

**Bộ dữ liệu hiện tại:** 32 ký tự × 5 mẫu = **160 mẫu**, gồm chữ số `0–9` và các chữ cái dùng trên biển số Việt Nam (không có `I`, `J`, `O`, `W`).

---

## 🎛️ Tham số có thể điều chỉnh

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|------------------|---------|
| `ADAPTIVE_THRESH_BLOCK_SIZE` | 19 | Kích thước vùng lân cận khi nhị phân hóa thích nghi |
| `ADAPTIVE_THRESH_WEIGHT` | 9 | Hằng số trừ khỏi giá trị trung bình khi nhị phân hóa |
| `GAUSSIAN_SMOOTH_FILTER_SIZE` | (5, 5) | Kích thước bộ lọc Gauss (càng lớn càng mờ) |
| `Min_char` / `Max_char` | 0.01 / 0.09 | Diện tích ký tự so với diện tích biển số |
| Tỷ lệ ký tự (w/h) | 0.25 – 0.7 | Khoảng tỷ lệ chấp nhận cho một ký tự |
| `RESIZED_IMAGE_WIDTH/HEIGHT` | 20 / 30 | Kích thước ảnh ký tự đưa vào KNN (phải khớp với `GenData.py`) |
| `k` (KNN) | 3 | Số láng giềng gần nhất |
| `MIN_CONTOUR_AREA` (`GenData.py`) | 40 | Diện tích contour tối thiểu khi tạo dữ liệu mẫu |
| `min_size_plate` / `max_size_plate` | 5000 / 18000 | Khai báo kích thước biển số (hiện chưa được sử dụng trong logic lọc) |

---

## 📊 Đánh giá kết quả

Trong video, **tỷ lệ phát hiện** được tính theo công thức:

```
tỷ lệ = (số biển số nhận dạng hợp lệ / tổng số khung hình) × 100%
```

Một số tiêu chí nên dùng để đo lường (có thể ghi vào `measurementCriteria.txt`):

- **Tỷ lệ phát hiện biển số** (detection rate).
- **Độ chính xác ký tự** (character accuracy) và **độ chính xác cả biển** (plate accuracy) so với nhãn thực tế.
- **Tốc độ xử lý** (FPS).
- Phân tích các ký tự hay bị nhầm (ví dụ `8/B`, `0/D`, `5/S`, `2/Z`).

---

## ⚠️ Hạn chế và hướng cải thiện

**Hạn chế hiện tại**

- `Image_test2.py` không lọc theo tỷ lệ w/h nên có thể nhận nhầm các vùng 4 cạnh khác là biển số.
- KNN với 160 mẫu dễ nhầm các ký tự có hình dạng tương tự; nhạy với góc chụp, ánh sáng và độ mờ.
- Chia dòng biển số dựa trên ngưỡng cố định `height / 3`.

**Hướng phát triển**

- Tăng số mẫu huấn luyện cho mỗi ký tự, bổ sung nhiều font và điều kiện ánh sáng.
- Dùng biến đổi phối cảnh (`getPerspectiveTransform`) thay cho xoay đơn giản.
- Đối chiếu kết quả với định dạng biển số Việt Nam để lọc kết quả sai.
- Thay KNN bằng CNN/SVM hoặc dùng YOLO để phát hiện biển số.
- Bỏ qua trùng lặp giữa các khung hình liên tiếp để đếm biển số chính xác hơn.

---

## 📚 Tài liệu tham khảo

1. [OpenCV Tutorial: Unlock the Power of Visual Data Processing – DataCamp](https://www.datacamp.com/tutorial/opencv-tutorial)
2. [OpenCV tutorial for beginners – FULL COURSE in 3 hours with Python (YouTube)](https://www.youtube.com/watch?v=eDIj5LuIL4A)
3. [Machine Learning – K-nearest neighbors (KNN) – W3Schools](https://www.w3schools.com/python/python_ml_knn.asp)
4. Các kênh YouTube tham khảo: Hà Nam 88, Computer Vision Engineer, Mrzaizai2k – AI, Nicholas Renotte, …