# tf_pca
PCA implementation in TensorFlow with support of Increamental PCA[[1]](#1). and Randomized SVD[[2]](#2).

## Example on TF Flowers dataset 
Data : https://www.tensorflow.org/datasets/catalog/tf_flowers

```python
from tfpca import PCA
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load tf flowers dataset
(ds_train, ds_test), dataset_info = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)

def normalize(img):
    min_val = tf.reduce_min(img)
    max_val = tf.reduce_max(img)
    return (img - min_val) / (max_val - min_val)

n_comp = 400
data_dim = (240,320,3)

# Label 0 is coltsfoot
ds_train = ds_train.filter(lambda x,y: y==0)
ds_test = ds_test.filter(lambda x,y: y==0)

# Resize data to 240, 320
data_train = ds_train.map(lambda x,y: tf.image.resize(x, data_dim[0:2]))
data_test = ds_test.map(lambda x,y: tf.image.resize(x, data_dim[0:2]))

X_train = list(data_train.batch(500, drop_remainder=True).take(1).as_numpy_iterator())[0]
X_test = list(data_test.batch(200, drop_remainder=True).take(1).as_numpy_iterator())[0]

# Train model
pca = PCA(n_comp, data_dim=data_dim, incremental=False, use_rSVD=True, p=10, q=1, dist= tf.random.normal)
pca.fit(X_train)

# Reconstruct
X_est = pca.reconstruct(X_train)

fig, axs = plt.subplots(1,2, figsize=(10,4))
[ax.axis('off') for ax in axs]
axs[0].imshow(normalize(X_est[0,...]))
axs[1].imshow(normalize(X_train[0,...]))
fig.suptitle("Reconstruction (Train)")
plt.show()

X_est = pca.reconstruct(X_test)

fig, axs = plt.subplots(1,2, figsize=(10,4))
[ax.axis('off') for ax in axs]
axs[0].imshow(normalize(X_est[0,...]))
axs[1].imshow(normalize(X_test[0,...]))
fig.suptitle("Reconstruction (Train)")
plt.show()
```
![image](https://github.com/ngunnar/tf_pca/assets/10964648/a5d04916-4061-4f4b-ba77-c458f56e6e33)
![image](https://github.com/ngunnar/tf_pca/assets/10964648/f6a722fa-a3a3-4136-8946-8269477ee43f)

## References
<a id="1">[1]</a> 
Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.

<a id="2">[2]</a> 
Ross, David A., et al. "Incremental learning for robust visual tracking." International journal of computer vision 77 (2008): 125-141.
