import numpy as np
import pandas as pd
from napari.layers import Image, Labels, Layer, Points, Surface, Vectors
from napari.types import LayerDataTuple


class TimelapseConverter:
    """
    Class to convert between 4D data and list of 3D data frames.

    Parameters
    ----------
    None

    Attributes
    ----------
    stack_data_functions : dict
        Dictionary with keys as supported data types and values as functions
        to convert list of 3D data frames to 4D data.
    unstack_data_functions : dict
        Dictionary with keys as supported data types and values as functions
        to convert 4D data to list of 3D data frames.
    supported_data : list
        List of supported data types.

    Methods
    -------
    unstack_data(data, layertype)
        Convert 4D data into a list of 3D data frames
        - Parameters
            data : 4D data to be converted
            layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
            `ImageData`, `LabelsData`, `List[LayerDataTuple]`, `LayerDataTuple` or
            pd.DataFrame.
    stack_data(data, layertype)
        Convert a list of 3D frames into 4D data.
        - Parameters
            data : list of 3D data (time)frames
            layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
            `ImageData`, `LabelsData`, `List[LayerDataTuple]`, `LayerDataTuple` or
            pd.DataFrame.
    """

    def __init__(self):

        self.stack_data_functions = {
            Points: self._stack_layer,
            Vectors: self._stack_layer,
            Surface: self._stack_layer,
            Image: self._stack_layer,
            Labels: self._stack_layer,
            Layer: self._stack_layer,
            LayerDataTuple: self._stack_layerdatatuple,
            "napari.types.PointsData": self._stack_points,
            "napari.types.VectorsData": self._stack_vectors,
            "napari.types.SurfaceData": self._stack_surfaces,
            "napari.types.ImageData": self._stack_image,
            "napari.types.LabelsData": self._stack_image,
            "napari.types.LayerDataTuple": self._stack_layerdatatuple,
        }

        self.unstack_data_functions = {
            Points: self._unstack_layer,
            Vectors: self._unstack_layer,
            Surface: self._unstack_layer,
            Image: self._unstack_layer,
            Labels: self._unstack_layer,
            Layer: self._unstack_layer,
            LayerDataTuple: self._unstack_layerdatatuple,
            "napari.types.PointsData": self._unstack_points,
            "napari.types.VectorsData": self._unstack_vectors,
            "napari.types.SurfaceData": self._unstack_surface,
            "napari.types.ImageData": self._unstack_image,
            "napari.types.LabelsData": self._unstack_image,
            "napari.types.LayerDataTuple": self._unstack_layerdatatuple,
        }

        self.supported_data = list(self.stack_data_functions.keys())

    def unstack_data(self, data, layertype: type) -> list:
        """
        Convert 4D data into a list of 3D data frames

        Parameters
        ----------
        data : 4D data to be converted
        layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
        `ImageData`, `LabelsData`, `List[LayerDataTuple]`, `LayerDataTuple` or
        pd.DataFrame.

        Raises
        ------
        TypeError
            Error to indicate that the converter does not support the passed
            layertype

        Returns
        -------
        list: List of 3D objects of input layertype

        """
        if layertype not in list(self.unstack_data_functions.keys()):
            raise TypeError(
                f"{layertype} data to list conversion currently not supported."
            )

        conversion_function = self.unstack_data_functions[layertype]
        return conversion_function(data)

    def stack_data(self, data, layertype: type):
        """
        Function to convert a list of 3D frames into 4D data.

        Parameters
        ----------
        data : list of 3D data (time)frames
        layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
        `ImageData`, `LabelsData`, `List[LayerDataTuple]`, `LayerDataTuple` or
        pd.DataFrame.

        Raises
        ------
        TypeError
            Error to indicate that the converter does not support the passed
            layertype

        Returns
        -------
        4D data of type `layertype`

        """
        if layertype not in self.supported_data:
            raise TypeError(
                f"{layertype} list to data conversion currently not supported."
            )
        conversion_function = self.stack_data_functions[layertype]
        return conversion_function(data)

    def _stack_image(self, images: list) -> "napari.types.ImageData":
        """Convert list of 3D images to single 4D image."""
        # Put images into a 4D array
        images = np.stack(images)
        return images

    def _unstack_image(self, image: "napari.types.ImageData") -> list:
        """Convert a 4D image to list of 3D images"""
        while len(image.shape) < 4:
            image = image[np.newaxis, :]
        return list(image)

    def _unstack_points(self, points: "napari.types.PointsData") -> list:
        """
        Function to convert a list of 3D frames into 4D data.

        Parameters
        ----------
        points : list of 3D data (time)frames

        Returns
        -------
        4D data of type `layertype`

        """
        # Accomodate for non-4D points (add zero-column if necessary)
        while points.shape[1] < 4:
            t = np.zeros(len(points), dtype=points.dtype)
            points = np.insert(points, 0, t, axis=1)

        n_frames = len(np.unique(points[:, 0]))
        points_list = [
            points[points[:, 0] == i][:, 1:] for i in range(n_frames)
        ]

        return points_list

    def _stack_points(self, points_list: list) -> "napari.types.PointsData":
        """
        Convert 4D data into a list of 3D data frames

        Parameters
        ----------
        points_list : list of 3D data

        Returns
        -------
        'napari.types.PointsData': 4D data

        """
        frame_column = [
            i * np.ones(len(points_list[i]), dtype=int)
            for i in range(len(points_list))
        ]
        points = np.concatenate(
            [
                np.column_stack((frame_column[i], points_list[i]))
                for i in range(len(points_list))
            ]
        )
        return points

    def _stack_vectors(self, vectors_list: list) -> "napari.types.VectorsData":
        """
        Convert a list of 3D frames into 4D data.

        Parameters
        ----------
        vectors_list : list of 3D data (time)frames

        Returns
        -------
        4D data of type `layertype`

        """
        base_points = [v[:, 0] for v in vectors_list]
        directions = [v[:, 1] for v in vectors_list]

        base_points = self._stack_points(base_points)
        directions = self._stack_points(directions)

        # Put result in (N, 2, 4) shape
        vectors = np.stack([base_points, directions]).transpose((1, 0, 2))
        return vectors

    def _unstack_vectors(self, vectors: "napari.types.VectorsData") -> list:

        base_points = vectors[:, 0]
        directions = vectors[:, 1]

        points_list = self._unstack_points(base_points)
        directions_list = self._unstack_points(directions)

        output_vectors = [
            np.stack([pt, vec]).transpose((1, 0, 2))
            for pt, vec in zip(points_list, directions_list)
        ]
        return output_vectors

    def _stack_surfaces(self, surfaces: list) -> tuple:
        """
        Convert list of 3D surfaces to single 4D surface.
        """
        # Put vertices, faces and values into separate lists
        vertices = [surface[0] for surface in surfaces]
        faces = [surface[1] for surface in surfaces]

        # Check if surfaces have a values entry
        if len(surfaces[0]) == 3:
            values = np.concatenate([surface[2] for surface in surfaces])
        else:
            values = None

        vertices = self._stack_points(vertices)

        # Calculate cumulative vertex counts for surfaces, adjust face indices
        # accordingly, and combine faces into one array.
        n_vertices = np.cumsum([surface[0].shape[0] for surface in surfaces])
        n_vertices = [0] + list(n_vertices)

        faces = [n_vertices[i] + face for i, face in enumerate(faces)]
        faces = np.vstack(faces)

        return (
            (vertices, faces, values)
            if values is not None
            else (vertices, faces)
        )

    def _unstack_surface(self, surface: "napari.types.SurfaceData") -> list:
        """Convert a 4D surface to list of 3D surfaces"""

        points = surface[0]
        faces = np.asarray(surface[1], dtype=int)

        # Check if values were assigned to the surface and ensure 4D points
        values = surface[2] if len(surface) == 3 else None
        points = np.pad(points, [(0, 0), (0, max(0, 4 - points.shape[1]))])

        # Get unique frames and points per frame
        unique_frames, points_per_frame = np.unique(
            points[:, 0], return_counts=True
        )

        # Determine at which index in the point array a new timeframe begins
        print(np.diff(points[faces[:, 0], 0]))
        idx_face_new_frame = (
            np.where(np.diff(points[faces[:, 0], 0]) != 0)[0] + 1
        )
        idx_face_new_frame = np.concatenate(
            ([0], idx_face_new_frame, [len(faces)])
        )

        # Initialize surfaces list
        surfaces = [None] * len(unique_frames)

        for t, (start, end) in enumerate(
            zip(idx_face_new_frame[:-1], idx_face_new_frame[1:])
        ):
            # Get points and faces for this frame
            frame_points = points[points[:, 0] == unique_frames[t], 1:]
            frame_faces = faces[start:end] - np.sum(points_per_frame[:t])

            # Get values for this frame, or use ones if no values were provided
            frame_values = (
                values[points[:, 0] == unique_frames[t]]
                if values is not None
                else np.ones(len(frame_points))
            )

            surfaces[t] = (frame_points, frame_faces, frame_values)

        return surfaces
    
    def _stack_layerdatatuple(self, layers: list) -> LayerDataTuple:
        """
        Convert list of 3D layerdatatuples to single 4D layerdatatuple.
        """

        layers = [Layer.create(*ldt) for ldt in layers]
        result = self._stack_layer(layers)
        
        return result.as_layer_data_tuple()
    
    def _unstack_layerdatatuple(self, layer: LayerDataTuple) -> list:
        """
        Convert a 4D layerdatatuple to list of 3D layerdatatuples
        """
        layer = Layer.create(*layer)
        result = self._unstack_layer(layer)
        
        return [layer.as_layer_data_tuple() for layer in result]


    def _stack_layer(self, layers: list) -> Layer:
        """
        Convert list of 3D layers to single 4D layer.
        """

        output_data = None
        output_features = None
        output_metadata = None

        # stack data based on layer type
        if isinstance(layers[0], Points):
            output_data = self._stack_points([layer.data for layer in layers])
        elif isinstance(layers[0], Vectors):
            output_data = self._stack_vectors([layer.data for layer in layers])
        elif isinstance(layers[0], Surface):
            output_data = self._stack_surfaces(
                [layer.data for layer in layers]
            )
        elif isinstance(layers[0], (Image, Labels)):
            output_data = self._stack_image([layer.data for layer in layers])

        # concatenate features, adding a 'frame' column to indicate the list index
        feature_list = [layer.features for layer in layers]
        output_features = pd.concat(
            [f.assign(frame=i) for i, f in enumerate(feature_list)]
        )

        # If metadata is present, concatenate it. Metadata is tyically a dictionary and cannot easily be handled by pandas
        metadata_list = [layer.metadata for layer in layers]
        metadata_values = {
            key: [metadata[key] for metadata in metadata_list]
            for key in list(metadata_list[0].keys())
        }
        output_metadata = {
            key: np.concatenate(metadata_values[key])
            for key in metadata_values
        }

        # create a new layer with the stacked data
        output_layer = type(layers[0])(output_data)
        output_layer.features = output_features
        output_layer.metadata = output_metadata

        return output_layer

    def _unstack_layer(self, layer: Layer) -> list:
        """
        Convert a 4D layer to list of 3D layers
        """
        # unstack data based on layer type
        time_column = None
        if isinstance(layer, Points):
            output_data = self._unstack_points(layer.data)
            time_column = layer.data[:, 0]

        elif isinstance(layer, Vectors):
            output_data = self._unstack_vectors(layer.data)
            time_column = layer.data[:, 0, 0]

        elif isinstance(layer, Surface):
            output_data = self._unstack_surface(layer.data)
            time_column = layer.data[0][:, 0]

        elif isinstance(layer, (Image, Labels)):
            output_data = self._unstack_image(layer.data)

        # unstack features
        if not hasattr(layer, "features"):
            feature_list = [pd.DataFrame() for _ in range(len(output_data))]

        elif layer.features is None or layer.features.empty:
            feature_list = [pd.DataFrame() for _ in range(len(output_data))]

        elif "frame" in layer.features.columns or time_column is None:
            feature_list = [
                layer.features[layer.features["frame"] == i]
                for i in range(len(output_data))
            ]
        else:
            feature_list = [
                layer.features[time_column == i]
                for i in np.unique(time_column)
            ]

        # unstack metadata
        n_frames = len(output_data)
        if layer.metadata is not None and layer.metadata != {}:
            metadata = (
                layer.metadata
            )  # this is a dict with n_frames entries per key
            output_metadata = [
                {key: metadata[key][i] for key in list(metadata.keys())}
                for i in range(n_frames)
            ]
        else:
            output_metadata = [{} for _ in range(n_frames)]

        # create a new layer with the unstacked data
        output_layers = [type(layer)(data) for data in output_data]
        for i, layer in enumerate(output_layers):
            layer.features = feature_list[i]
            layer.metadata = output_metadata[i]

        return output_layers
