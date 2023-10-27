import argparse
import cv2
import json
from matplotlib import pyplot
import numpy
from pathlib import Path

from sklearn.neighbors import NearestNeighbors

from image_regressor.vis import highlight_images


# Size that is good for downsampled viewing of many images, on the order of
# 3x5 stacked images or so
DOWNSIZE = numpy.array([335, 253])


# TODO: Try different distances: two best candidates are euclidean and
# cosine. Pass into NearestNeighbors as 'metric'
def inspect_knn(embeddings, imnames, sample, k, imdir, savedir):

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embeddings)

    samples = numpy.random.choice(range(len(imnames)), size=sample, replace=False)
    # Whenever we use the distances or indices we must skip the first one, that
    # is the original point
    sample_dists, sample_indices = nn.kneighbors(embeddings[samples])

    for i, (index, k_dists, k_indices) in enumerate(
        zip(samples, sample_dists, sample_indices)
    ):
        highlight_images(
            hero=imdir.joinpath(imnames[index]),
            others=[imdir.joinpath(imnames[k_index]) for k_index in k_indices[1:]],
            path=savedir.joinpath(f"knn_sample_{i:02}.jpg"),
            size=DOWNSIZE,
        )
    print(f"Saved {sample} {k}-nn images to {savedir}")


def inspect_differentials(embeddings, imnames, sample, k, imdir, savedir):
    """
    Sample randomly X times and then show oher random samples but sorted so
    that the viewer can see the respective distances
    """

    samples = numpy.random.choice(range(len(imnames)), size=sample, replace=False)

    # TODO: Build in cosine distance capability
    for i, index in enumerate(samples):

        # Check a feasibly large number of vectors, select the largest distance
        # and also the closest to 0.5
        subsamples = numpy.random.choice(range(len(imnames)), size=k, replace=False)
        distances = numpy.linalg.norm(embeddings[subsamples] - embeddings[index], axis=1)

        # Get the max and the closest to 50%
        maxind = distances.argmax()
        maxval = distances.max()
        midind = abs((distances / maxval) - 0.5).argmin()
        midval = distances[midind]
        lowind = abs((distances / maxval) - 0.2).argmin()
        lowval = distances[lowind]

        highlight_images(
            hero=imdir.joinpath(imnames[index]),
            others=[imdir.joinpath(imnames[subsamples[ind]]) for ind in (lowind, midind, maxind)],
            path=savedir.joinpath(f"differential_sample_{i:02}.jpg"),
            size=DOWNSIZE,
            pad=20,
            text=[f"{lowval:.3}", f"{midval:.3}", f"{maxval:.3}"],
        )

        # Also save a histogram of the distance distributions
        pyplot.hist(x=distances)
        pyplot.xlabel("Distance from sample")
        pyplot.ylabel("Count")
        pyplot.title(f"Histogram for distance from {imnames[index]}")
        pyplot.tight_layout()
        pyplot.savefig(savedir.joinpath(f"differential_sample_{i:02}_hist.jpg"))
        pyplot.close()

    print(f"Saved {sample} differential images to {savedir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create report visualizations for embedding quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--embedding-file",
        help="json dict with paired image:embedding in the 'data' field",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-i",
        "--image-directory",
        help="Directory with all images we want to examine",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--save-directory",
        help="Directory to store output files in",
        required=True,
        type=Path,
    )
    args = parser.parse_args()
    assert args.embedding_file.is_file()
    assert args.image_directory.is_dir()
    assert args.save_directory.is_dir()

    data = json.load(args.embedding_file.open("r"))["data"]
    imnames = sorted(data.keys())
    embeddings = numpy.array([data[key] for key in imnames])

    # inspect_knn(
    #     embeddings,
    #     imnames,
    #     sample=10,
    #     k=4,
    #     imdir=args.image_directory,
    #     savedir=args.save_directory,
    # )

    inspect_differentials(
        embeddings,
        imnames,
        sample=10,
        k=2000,
        imdir=args.image_directory,
        savedir=args.save_directory,
    )
