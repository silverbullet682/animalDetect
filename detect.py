import math
import os
import pickle
import cv2
from sklearn.cluster import KMeans


def load_image(path_image):
    # Đọc ảnh với hàm imread OpenCV
    img = cv2.imread(path_image)
    return img


def read_data(path_to_image):
    # tạo mảng
    X = []  # chứa image
    Y = []  # chứa nhãn(label)
    # os.listdir Trả về một danh sách chứa các tên của các entry trong thư mục đã cho bởi path.
    # os.path.join : nối đường dẫn
    for label in os.listdir(path_to_image):
        for img_file in os.listdir(os.path.join(path_to_image, label)):
            # load ảnh từ đường dẫn
            img = load_image(os.path.join(path_to_image, label, img_file))
            # thêm ảnh vào mảng X
            X.append(img)
            # thêm nhãn vào mảng Y
            Y.append(label)
    # trả về mảng X chứa image, mảng Y chứa nhãn
    return X, Y


def extract_sift_features(X):
    image_descriptors = []
    # khởi tạo thuật toán SIFT
    sift = cv2.SIFT_create()
    # duyệt tất cả các ảnh
    for i in range(len(X)):
        # chuyển ảnh sang màu xám
        gray_image = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
        # trích rút đặc trưng, thu về mảng keypoint và descriptors
        key, des = sift.detectAndCompute(gray_image, None)
        # lưu descrriptors của mỗi ảnh vào mảng image_descriptors
        image_descriptors.append(des)
    return image_descriptors


def kmeans_cluster(image_descriptors, cluster):
    # Hàm Kmeans của thư viện sklearn
    kmeans = KMeans(n_clusters=cluster, init='random', max_iter=100).fit(image_descriptors)
    # lưu mảng nhãn vào biến labels
    labels = kmeans.labels_
    return labels


def create_features(image_descriptors, num_cluster):
    X_features = []
    # duyệt descriptor của mỗi ảnh
    for i in range(0, len(image_descriptors)):
        # lưu vecto đặc trưng của mỗi ảnh vào mảng X_feature
        # hàm bag_of_feature để chuẩn hóa ra vecto đặc trưng của ảnh
        X_features.append(bag_of_feature(image_descriptors[i], num_cluster))
    return X_features


def bag_of_feature(image_des, num_clusters):
    # hàm kmeans_cluster trả về danh sách nhãn của các descriptor
    label = kmeans_cluster(image_des, num_clusters)
    # khởi tạo hàm count với num_clusters phần tử
    count = list(range(num_clusters))
    # Khởi tạo tất cả giá trị phần tử của mảng = 0
    for i in range(0, num_clusters):
        count[i] = 0
    for j in range(0, len(label)):
        count[label[j]] += 1
    for j in range(0, num_clusters):
        count[j] = count[j] / len(image_des)
    return count


def calc_distance(item, point):
    dis = 0
    # áp dụng euclid tính khoảng cách
    for i in range(0, len(item)):
        dis += pow(item[i] - point[i], 2)

    return math.sqrt(dis)


def find_most(arr):
    # tìm nhãn xuất hiện nhiều nhất trong mảng
    labels = set(arr)
    ans = ""
    max_occur = 0
    for label in labels:
        num = arr.count(label)
        if num > max_occur:
            max_occur = num
            ans = label
    return ans


def k_nearest(training_set, point, k, label):
    # tìm k phần tử gần nhất
    distances = []
    for i in range(0, len(training_set)):
        distances.append({
            "label": label[i],  # lấy nhãn
            "value": calc_distance(training_set[i], point)  # tính khoảng cách
        })
    # sắp xếp theo khoảng cách
    distances.sort(key=lambda x: x["value"])
    # lấy nhãn chuyển sang 1 mảng
    labels = [item["label"] for item in distances]
    return labels[:k]


# đọc dữ liệu ở file dpt
X, Y = read_data('dpt')
num_clusters = 20
if not os.path.isfile('bow_dictionary.pkl'):
    # trích rút đặc trưng bằng thuật toán sift
    des = extract_sift_features(X)
    # mảng vector đặc trưng của bộ train
    training = create_features(des, num_clusters)
    # lưu dữ liệu training vào file
    pickle.dump(training, open('bow_dictionary.pkl', 'wb'))
else:
    # đọc dữ liệu training từ file
    training = pickle.load(open('bow_dictionary.pkl', 'rb'))

# đọc ảnh cần kiểm tra nhãn
test_image = cv2.imread('3.png')
# khởi tạo thuật toán Sift
sift = cv2.SIFT_create()
# chuyển ảnh sang màu xám
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# Trích rút đặc trưng
key, des = sift.detectAndCompute(gray_image, None)
# Chuẩn hóa đặc trưng thành 1 vector đặc trưng
feauture_test = bag_of_feature(des, num_clusters)
# Dùng KNN để tìm nhãn
# Trả về mảng k phần tử gồm giá trị khoảng cách và nhãn tương ứng
knn = k_nearest(training, feauture_test, 11, Y)
# tìm nhãn xuất hiện nhiều nhất trong mảng
result = find_most(knn)
#in kết quả nhãn
print("Dự đoán: "+result)
