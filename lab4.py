import numpy as np

from sklearn.cluster import KMeans

train_samples_amount = 60000
train_images_filepath = "/home/maksim/Desktop/ml/mnist/train-images.idx3-ubyte"
train_labels_filepath = "/home/maksim/Desktop/ml/mnist/train-labels.idx1-ubyte"

test_samples_amount = 10000
test_images_filepath = "/home/maksim/Desktop/ml/mnist/t10k-images.idx3-ubyte"
test_labels_filepath = "/home/maksim/Desktop/ml/mnist/t10k-labels.idx1-ubyte"

epoch_amount = 5

img_width = 28
img_height = 28

def read_mnist(images_path, labels_path, samples_amount):
    images = []
    labels = []
    with open(images_path, "rb") as f:
        f.read(16)
        for k in range(samples_amount):
            curr_image = []
            for i in range(img_height):
                for j in range(img_width):
                    number = f.read(1)
                    if number == b'\x00':
                        curr_image.append(0)
                    else:
                        curr_image.append(1)
            images.append(curr_image)
    with open(labels_path, "rb") as f:
        f.read(8)
        for k in range(samples_amount):
            label = f.read(1)
            labels.append(int.from_bytes(label, byteorder='little'))
    return np.array(images), np.array(labels)

def print_img(image):
    for i in range(img_height):
        row = ""
        for j in range(img_width):
            row += str(image[i * img_width + j])
        print(row)

def get_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# def kmeans(xs, k, max_iters=1000):
#     conv = False
#     centroids = xs[np.random.choice(range(len(xs)), size=k, replace=False)]
    
#     it = 0
#     while (not conv) and it < max_iters:
#         clusters = [[] for _ in range(len(centroids))]
#         for x in xs:
#             min_dist = np.inf
#             closest_c = -1
#             for j, c in enumerate(centroids):
#                 dist = get_distance(x, c)
#                 if dist < min_dist:
#                     min_dist = dist
#                     closest_c = j
#             clusters[closest_c].append(x)
#         clusters = [c for c in clusters if c]
#         prev_centroids = centroids.copy()
#         centroids = [np.mean(c, axis=0) for c in clusters]
        
#         conv = len(prev_centroids) == len(centroids) and np.allclose(prev_centroids, centroids)
#         it += 1
            
#     return np.array(centroids), [np.std(x) for x in clusters]

def kmeans(xs, k, max_iters):
    centroids = KMeans(n_clusters=k, max_iter=max_iters).fit(xs).cluster_centers_

    max_d = -1
    for i in range(len(centroids)):
        for j in range(i, len(centroids)):
            max_d = max(get_distance(centroids[i], centroids[j]), max_d)
    return np.array(centroids), [max_d for c in centroids]


class RBF:

    def __init__(self, X, y, tX, ty, num_of_classes,
                 k, std_from_clusters=True):
        self.train_X = X
        self.train_y = y

        self.test_X = tX
        self.test_y = ty

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    def convert_label_to_ans(self, x, num_of_classes): #array [0, 0, 1, 0, 0, 0 ,0, 0, 0, 0]
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(distance ** 2/ s ** 2)

    def rbf_list(self, X, centroids, std_list):
        xs_clusters_distrib = []
        for x in X:
            clusters_distrib = []
            for c, s in zip(centroids, std_list):
                clusters_distrib.append(self.rbf(x, c, s))
            xs_clusters_distrib.append(np.array(clusters_distrib))
        return np.array(xs_clusters_distrib)
    
    def fit(self):

        self.centroids, self.std_list = kmeans(self.train_X, self.k, max_iters=1000)
        print("kmeans computed")

        if not self.std_from_clusters:
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        RBF_X = self.rbf_list(self.train_X, self.centroids, self.std_list)

        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_label_to_ans(self.train_y, self.number_of_classes)
        print("weights computed")

    def test(self):
        RBF_list_tst = self.rbf_list(self.test_X, self.centroids, self.std_list)

        self.pred_ty = RBF_list_tst @ self.w

        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        diff = self.pred_ty - self.test_y

        return len(np.where(diff == 0)[0]) / len(diff)


def main():
    train_images, train_lables = read_mnist(train_images_filepath, train_labels_filepath, train_samples_amount)
    test_images, test_labels = read_mnist(test_images_filepath, test_labels_filepath, test_samples_amount)
    print('data loaded')
    res = []
    for k in [3, 5, 10, 50, 100]:
        RBF_NN = RBF(train_images, train_lables, test_images, test_labels, num_of_classes=10,
                     k=k, std_from_clusters=True)
        RBF_NN.fit()
        acc = RBF_NN.test()
        res.append((k, acc))
        print(f'{k}: accuracy = {acc}')
    for k, acc in res:
        print(f'{k}: accuracy = {acc}')

main()

# 3: accuracy = 0.3031
# 5: accuracy = 0.5122
# 10: accuracy = 0.6598
# 50: accuracy = 0.8977
# 100: accuracy = 0.9198
