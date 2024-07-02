import inspect
from functools import wraps

import tqdm
from dask.distributed import Client, get_client

from .timelapse_converter import TimelapseConverter


def frame_by_frame(function: callable, progress_bar: bool = True):
    """
    Decorator to apply a function frame by frame to 4D data.

    Parameters
    ----------
    function : callable
        Function to be wrapped. If the optional argument `use_dask` is passed
        to the function, the function will be parallelized using dask:

        >>> @frame_by_frame(some_function)(argument1, argument2, use_dask=True)

        *Note*: For this to work, the arguments (e.g., the input data) must not be passed as keyword
        argument. I.e., this works:

        >>> @frame_by_frame(some_function)(argument1, argument2, some_keyword='abc', use_dask=True)

        This does not work:

        >>> @frame_by_frame(some_function)(image1=argument1, image2=argument2, some_keyword='abc', use_dask=True)

    progress_bar : bool, optional
        Show progress bar, by default False. Has no effect if `use_dask=True` is passed as an argument
        to the input function `function`.

    Returns
    -------
    callable
        Wrapped function
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(function)
        annotations = [
            sig.parameters[key].annotation for key in sig.parameters
        ]

        converter = TimelapseConverter()

        args = list(args)
        n_frames = None

        # Inspect arguments and check if `use_dask` is passed as keyword argument
        use_dask = False
        if "use_dask" in kwargs:
            use_dask = kwargs["use_dask"]
            del kwargs["use_dask"]

        # Convert 4D data to list(s) of 3D data for every supported argument
        # and store the list in the same place as the original 4D data
        index_of_converted_arg = []  # remember which arguments were converted

        for idx, arg in enumerate(args):
            if annotations[idx] in converter.supported_data:
                args[idx] = converter.unstack_data(arg, annotations[idx])
                index_of_converted_arg.append(idx)
                n_frames = len(args[idx])

        # apply function frame by frame
        results = [None] * n_frames
        frames = (
            tqdm.tqdm(range(n_frames), leave=False)
            if progress_bar
            else range(n_frames)
        )

        # start dask cluster client
        if use_dask:
            try:
                client = get_client()
                print(
                    "Dask client already running",
                    client,
                    f" Log: {client.dashboard_link}",
                )
            except ValueError:
                client = Client()
                print(
                    "Dask client up and running",
                    client,
                    f" Log: {client.dashboard_link}",
                )
            jobs = []

        for t in frames:
            _args = args.copy()

            # Replace 4D argument by single frame (arg[t])
            for idx in index_of_converted_arg:
                _args[idx] = _args[idx][t]

            if use_dask:
                # args_futures = [client.scatter(arg) for arg in _args]
                jobs.append(client.submit(function, *_args, **kwargs))
            else:
                results[t] = function(*_args, **kwargs)

        if use_dask:
            # gather results
            results = client.gather(jobs)

        return converter.stack_data(results, sig.return_annotation)

    return wrapper
