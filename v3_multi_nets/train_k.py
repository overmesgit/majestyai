from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from v3_multi_nets.areas import Hero

room_data = [
    [Hero(100, 1, 1), Hero(-20, 0.1, 1), 0.4],
    [Hero(100, 1, 1), Hero(-50, 0.1, 1), 0.3],
    [Hero(100, 1, 1), Hero(-70, 0.1, 1), 0.2],
    [Hero(80, 1, 1), Hero(-40, 0.1, 1), 0.2],
    [Hero(80, 1, 1), Hero(-60, 0.1, 1), 0.1],
    [Hero(80, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-40, 0.1, 1), 0.1],
    [Hero(60, 1, 1), Hero(-60, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    [Hero(60, 1, 1), Hero(-80, 0, 0), 0],
    # heal
    [Hero(80, 1, 1), Hero(20, 0, 0), 0.3],
    [Hero(60, 1, 1), Hero(60, 0, 0), 0.35],
    [Hero(20, 1, 1), Hero(80, 0, 0), 0.4],
    [Hero(100, 1, 1), Hero(0, 0, 0), 0],
    [Hero(80, 1, 1), Hero(20, 0, 0), 0.1],

]


# Create a MiniBatchKMeans instance
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=5)

# Use partial_fit
data = [d.to_array() for (h, d, _) in room_data]

for i in range(0, len(data), 5):
    kmeans.partial_fit(data[i:i+5])

for d in data:
    labels = kmeans.predict([d])
    print(d, labels)
