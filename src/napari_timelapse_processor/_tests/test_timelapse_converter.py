import numpy as np


def test_convert_images(create_3D_image, create_4D_image):
    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D images to 4d images and back
    list_of_3d_images = [create_3D_image for _ in range(5)]
    converted_images = Converter.stack_data(
        list_of_3d_images, layertype="napari.types.ImageData"
    )

    back_converted = Converter.unstack_data(
        converted_images, layertype="napari.types.ImageData"
    )

    for i in range(5):
        assert np.array_equal(list_of_3d_images[i], back_converted[i])

    # Now we check conversion from 4d images to list of 3d images and back
    images_4d = create_4D_image
    converted_images = Converter.unstack_data(
        images_4d, layertype="napari.types.ImageData"
    )
    back_converted = Converter.stack_data(
        converted_images, layertype="napari.types.ImageData"
    )

    assert np.array_equal(images_4d, back_converted)


def test_convert_labels(create_3D_labels, create_4D_labels):
    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D labels to 4d labels and back
    list_of_3d_labels = [create_3D_labels for _ in range(5)]
    converted_labels = Converter.stack_data(
        list_of_3d_labels, layertype="napari.types.LabelsData"
    )

    back_converted = Converter.unstack_data(
        converted_labels, layertype="napari.types.LabelsData"
    )

    for i in range(5):
        assert np.array_equal(list_of_3d_labels[i], back_converted[i])

    # Now we check conversion from 4d labels to list of 3d labels and back
    labels_4d = create_4D_labels
    converted_labels = Converter.unstack_data(
        labels_4d, layertype="napari.types.LabelsData"
    )
    back_converted = Converter.stack_data(
        converted_labels, layertype="napari.types.LabelsData"
    )

    assert np.array_equal(labels_4d, back_converted)


def test_convert_points(create_3D_points, create_4D_points):
    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D points to 4d points and back
    list_of_3d_points = [create_3D_points for _ in range(5)]
    converted_points = Converter.stack_data(
        list_of_3d_points, layertype="napari.types.PointsData"
    )

    back_converted = Converter.unstack_data(
        converted_points, layertype="napari.types.PointsData"
    )

    for i in range(5):
        assert np.array_equal(list_of_3d_points[i], back_converted[i])

    # Now we check conversion from 4d points to list of 3d points and back
    points_4d = create_4D_points
    converted_points = Converter.unstack_data(
        points_4d, layertype="napari.types.PointsData"
    )
    back_converted = Converter.stack_data(
        converted_points, layertype="napari.types.PointsData"
    )

    assert np.array_equal(points_4d, back_converted)


def test_convert_points_with_features(
    create_3dpoints_layer_with_features, create_4dpoints_layer_with_features
):
    from napari.layers import Points

    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from 3D points with features to 4d points with features and back
    list_of_3d_points = [create_3dpoints_layer_with_features for _ in range(5)]
    converted_points = Converter.stack_data(
        list_of_3d_points, layertype=Points
    )

    back_converted = Converter.unstack_data(converted_points, layertype=Points)

    for i in range(5):
        assert np.array_equal(
            list_of_3d_points[i].data, back_converted[i].data
        )
        for col in list_of_3d_points[i].features.columns:
            assert np.array_equal(
                list_of_3d_points[i].features[col].values,
                back_converted[i].features[col].values,
            )

    # Now we check conversion from 4d points with features to 3d points with features and back
    points_4d = create_4dpoints_layer_with_features
    converted_points = Converter.unstack_data(points_4d, layertype=Points)
    back_converted = Converter.stack_data(converted_points, layertype=Points)

    assert np.array_equal(points_4d.data, back_converted.data)
    for col in points_4d.features.columns:
        if col == "frame":
            continue
        assert np.array_equal(
            points_4d.features[col].values, back_converted.features[col].values
        )


def test_convert_vectors(create_3D_vectors, create_4D_vectors):
    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D vectors to 4d vectors and back
    list_of_3d_vectors = [create_3D_vectors for _ in range(5)]
    converted_vectors = Converter.stack_data(
        list_of_3d_vectors, layertype="napari.types.VectorsData"
    )

    back_converted = Converter.unstack_data(
        converted_vectors, layertype="napari.types.VectorsData"
    )

    for i in range(5):
        assert np.array_equal(list_of_3d_vectors[i], back_converted[i])

    # Now we check conversion from 4d vectors to list of 3d vectors and back
    vectors_4d = create_4D_vectors
    converted_vectors = Converter.unstack_data(
        vectors_4d, layertype="napari.types.VectorsData"
    )
    back_converted = Converter.stack_data(
        converted_vectors, layertype="napari.types.VectorsData"
    )

    assert np.array_equal(vectors_4d, back_converted)


def test_convert_mesh(create_3d_mesh, create_4d_mesh):
    from napari_timelapse_processor import TimelapseConverter

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D mesh to 4d mesh and back
    list_of_3d_mesh = [create_3d_mesh for _ in range(5)]
    converted_mesh = Converter.stack_data(
        list_of_3d_mesh, layertype="napari.types.SurfaceData"
    )

    back_converted = Converter.unstack_data(
        converted_mesh, layertype="napari.types.SurfaceData"
    )

    [print(back_converted[i][0].shape) for i in range(5)]
    [print(list_of_3d_mesh[i][0].shape) for i in range(5)]

    for i in range(5):
        assert np.array_equal(list_of_3d_mesh[i][0], back_converted[i][0])
        assert np.array_equal(list_of_3d_mesh[i][1], back_converted[i][1])

    # Now we check conversion from 4d mesh to list of 3d mesh and back
    mesh_4d = create_4d_mesh
    converted_mesh = Converter.unstack_data(
        mesh_4d, layertype="napari.types.SurfaceData"
    )
    back_converted = Converter.stack_data(
        converted_mesh, layertype="napari.types.SurfaceData"
    )

    assert np.array_equal(mesh_4d[0], back_converted[0])
    assert np.array_equal(mesh_4d[1], back_converted[1])

def test_faces_index_after_stacking():
    from napari_timelapse_processor import TimelapseConverter
    Converter = TimelapseConverter()
    # Define mock surfaces with vertices and faces
    surfaces = [
        (
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            np.array([[0, 1, 2]])),  # 3 vertices, 1 face
        (
            np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]]),
            np.array([[0, 1, 2]])),  # 3 vertices, 1 face
    ]

    # Stack surfaces
    result = Converter._stack_surfaces(surfaces)

    # The lowest index in the faces array at timestep 1 should be 3,
    # as there are 3 vertices (0, 1, 2) before it
    assert result[1][0].min() == 0
    assert result[1][0].max() == 2
    assert result[1][1].min() == 3
    assert result[1][1].max() == 5

# def create_3d_layerdatatuple_with_features():
#     import pandas as pd

#     points = np.random.random((10, 3))
#     features = pd.DataFrame(
#         {
#             "feature1": np.random.random(len(points)),
#             "feature2": np.random.random(len(points)),
#         }
#     )

#     layerdatatuple = (
#         points,
#         {'features': features,
#          'name': 'pointcloud',
#         },
#         'points')
             
#     return layerdatatuple

def test_convert_layerdatatuple(create_3d_layerdatatuple_with_features):
    from napari_timelapse_processor import TimelapseConverter
    import pandas as pd

    Converter = TimelapseConverter()

    # First we check conversion from list of 3D layerdatatuple to 4d layerdatatuple and back
    list_of_3d_layerdatatuple = [create_3d_layerdatatuple_with_features for _ in range(5)]
    converted_layerdatatuple = Converter.stack_data(
        list_of_3d_layerdatatuple, layertype="napari.types.LayerDataTuple"
    )

    back_converted = Converter.unstack_data(
        converted_layerdatatuple, layertype="napari.types.LayerDataTuple"
    )

    for i in range(5):
        assert np.array_equal(
            list_of_3d_layerdatatuple[i][0], back_converted[i][0]
        )
        features1 = list_of_3d_layerdatatuple[i][1]['features']
        features2 = back_converted[i][1]['features'].drop('frame', axis=1)
        assert features1.equals(features2)

if __name__ == "__main__":
    test_convert_layerdatatuple(create_3d_layerdatatuple_with_features)