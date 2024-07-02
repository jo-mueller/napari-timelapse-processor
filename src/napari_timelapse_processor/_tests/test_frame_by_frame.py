def image_processing_function(
    image: "napari.types.ImageData",
) -> "napari.types.ImageData":
    return image * 2


def label_processing_function(
    labels: "napari.types.LabelsData",
) -> "napari.types.LabelsData":
    return labels + 1


def points_processing_function(
    points: "napari.types.PointsData",
) -> "napari.types.PointsData":
    return points * 2


def vectors_processing_function(
    vectors: "napari.types.VectorsData",
) -> "napari.types.VectorsData":
    return vectors * 2


def mesh_processing_function(
    mesh: "napari.types.SurfaceData",
) -> "napari.types.SurfaceData":
    vertices = mesh[0] * 2
    return (vertices, mesh[1])


def test_frame_by_frame_image(create_4D_image):
    from napari_timelapse_processor import frame_by_frame

    image = create_4D_image
    processed_image = frame_by_frame(image_processing_function)(image)
    assert processed_image.shape == image.shape
    assert (processed_image == image * 2).all()

    # try in distributed mode
    processed_image = frame_by_frame(image_processing_function)(
        image, use_dask=True
    )
    assert processed_image.shape == image.shape
    assert (processed_image == image * 2).all()


def test_frame_by_frame_labels(create_4D_labels):
    from napari_timelapse_processor import frame_by_frame

    labels = create_4D_labels
    processed_labels = frame_by_frame(label_processing_function)(labels)
    assert processed_labels.shape == labels.shape
    assert (processed_labels[0] == labels[0] + 1).all()
    assert (processed_labels[1] == labels[1] + 1).all()

    # try in distributed mode
    processed_labels = frame_by_frame(label_processing_function)(
        labels, use_dask=True
    )
    assert processed_labels.shape == labels.shape
    assert (processed_labels[0] == labels[0] + 1).all()
    assert (processed_labels[1] == labels[1] + 1).all()


def test_frame_by_frame_points(create_4D_points):
    from napari_timelapse_processor import frame_by_frame

    points = create_4D_points
    processed_points = frame_by_frame(points_processing_function)(points)
    assert processed_points.shape == points.shape
    assert (processed_points[:, 1:] == points[:, 1:] * 2).all()


def test_frame_by_frame_vectors(create_4D_vectors):
    from napari_timelapse_processor import frame_by_frame

    vectors = create_4D_vectors
    processed_vectors = frame_by_frame(vectors_processing_function)(vectors)
    assert processed_vectors.shape == vectors.shape
    assert (processed_vectors[:, 0, 1:] == vectors[:, 0, 1:] * 2).all()
    assert (processed_vectors[:, 1, 1:] == vectors[:, 1, 1:] * 2).all()

    # try in distributed mode
    processed_vectors = frame_by_frame(vectors_processing_function)(
        vectors, use_dask=True
    )
    assert processed_vectors.shape == vectors.shape
    assert (processed_vectors[:, 0, 1:] == vectors[:, 0, 1:] * 2).all()
    assert (processed_vectors[:, 1, 1:] == vectors[:, 1, 1:] * 2).all()


def test_frame_by_frame_mesh(create_4d_mesh):
    from napari_timelapse_processor import frame_by_frame

    mesh = create_4d_mesh
    processed_mesh = frame_by_frame(mesh_processing_function)(mesh)
    assert processed_mesh[0].shape == mesh[0].shape
    assert processed_mesh[1].shape == mesh[1].shape
    assert (processed_mesh[0][:, 1:] == mesh[0][:, 1:] * 2).all()
    assert (processed_mesh[1] == mesh[1]).all()

    # try in distributed mode
    processed_mesh = frame_by_frame(mesh_processing_function)(
        mesh, use_dask=True
    )
    assert processed_mesh[0].shape == mesh[0].shape
    assert processed_mesh[1].shape == mesh[1].shape
    assert (processed_mesh[0][:, 1:] == mesh[0][:, 1:] * 2).all()
    assert (processed_mesh[1] == mesh[1]).all()
