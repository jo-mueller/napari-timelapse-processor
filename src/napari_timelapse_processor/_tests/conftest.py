import numpy as np
import pytest


@pytest.fixture
def create_3D_image():
    image = np.random.random((10, 10, 10))
    return image


@pytest.fixture
def create_4D_image():
    n_timepoints = 5
    image = np.random.random((n_timepoints, 10, 10, 10))
    return image


@pytest.fixture
def create_3D_labels():
    labels = np.zeros((10, 10, 10))
    labels[4:6, 4:6, 4:6] = 1
    labels[7:9, 7:9, 7:9] = 2
    return labels


@pytest.fixture
def create_4D_labels():
    n_timepoints = 5
    labels = np.zeros((n_timepoints, 10, 10, 10))
    labels[0, 4:6, 4:6, 4:6] = 1
    labels[1, 7:9, 7:9, 7:9] = 2
    return labels


@pytest.fixture
def create_3D_points():
    points = np.random.random((10, 3))
    return points


@pytest.fixture
def create_4D_points():
    n_timepoints = 5
    points = np.random.random((100, 4))
    # first column is time axis
    points[:, 0] = np.repeat(np.arange(n_timepoints), 100 // n_timepoints)

    return points


@pytest.fixture
def create_3D_vectors():
    vectors = np.random.random((10, 2, 3))
    return vectors


@pytest.fixture
def create_4D_vectors():
    n_timepoints = 5
    vectors = np.random.random((100, 2, 4))
    # first column is time axis
    vectors[:, 0, 0] = np.repeat(np.arange(n_timepoints), 100 // n_timepoints)
    vectors[:, 1, 0] = np.repeat(np.arange(n_timepoints), 100 // n_timepoints)

    return vectors


@pytest.fixture
def create_3d_mesh():

    vertices = np.random.random((10, 3))
    faces = np.random.randint(0, 10, (10, 3))

    return (vertices, faces)


@pytest.fixture
def create_4d_mesh():

    n_timepoints = 5
    vertices = np.random.random((100, 4))
    vertices[:, 0] = np.repeat(np.arange(n_timepoints), 100 // n_timepoints)

    points_per_timepoint = 100 // n_timepoints

    # faces must only connect vertices of the same timepoint
    faces = np.random.randint(0, points_per_timepoint, (10, 3))
    faces = np.stack(
        [
            face + i * points_per_timepoint
            for i in range(n_timepoints)
            for face in faces
        ]
    )

    return (vertices, faces)


@pytest.fixture
def create_3dpoints_layer_with_features(create_3D_points):
    import pandas as pd
    from napari.layers import Points

    features = pd.DataFrame(
        {"feature1": np.random.random(10), "feature2": np.random.random(10)}
    )

    layer = Points(create_3D_points, features=features)
    return layer


@pytest.fixture
def create_4dpoints_layer_with_features(create_4D_points):
    import pandas as pd
    from napari.layers import Points

    features = pd.DataFrame(
        {
            "feature1": np.random.random(len(create_4D_points)),
            "feature2": np.random.random(len(create_4D_points)),
        }
    )

    layer = Points(create_4D_points, features=features)
    return layer


@pytest.fixture
def create_3dvectors_layer_with_features():
    import pandas as pd
    from napari.layers import Vectors

    vectors = create_3D_vectors()
    features = pd.DataFrame(
        {"feature1": np.random.random(10), "feature2": np.random.random(10)}
    )

    layer = Vectors(vectors, features=features)
    return layer


@pytest.fixture
def create_4dvectors_layer_with_features():
    import pandas as pd
    from napari.layers import Vectors

    vectors = create_4D_vectors()
    features = pd.DataFrame(
        {"feature1": np.random.random(100), "feature2": np.random.random(100)}
    )

    layer = Vectors(vectors, features=features)
    return layer


@pytest.fixture
def create_3dmesh_layer_with_features():
    import pandas as pd
    from napari.layers import Surface

    vertices, faces = create_3d_mesh()
    features = pd.DataFrame(
        {
            "feature1": np.random.random(len(vertices)),
            "feature2": np.random.random(len(vertices)),
        }
    )

    layer = Surface((vertices, faces))
    layer.features = features
    return layer


@pytest.fixture
def create_4dmesh_layer_with_features():
    import pandas as pd
    from napari.layers import Surface

    vertices, faces = create_4d_mesh()
    features = pd.DataFrame(
        {
            "feature1": np.random.random(len(vertices)),
            "feature2": np.random.random(len(vertices)),
        }
    )

    layer = Surface((vertices, faces))
    layer.features = features
    return layer


@pytest.fixture
def create_3d_layerdatatuple_with_features():
    import pandas as pd

    points = np.random.random((10, 3))
    features = pd.DataFrame(
        {
            "feature1": np.random.random(len(points)),
            "feature2": np.random.random(len(points)),
        }
    )

    layerdatatuple = (
        points,
        {'features': features,
         'name': 'pointcloud',
        },
        'points')
             
    return layerdatatuple