import argparse
import cv2
import json
from matplotlib import pyplot
import numpy
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from image_regressor.vis import highlight_images


# Size that is good for downsampled viewing of many images, on the order of
# 3x5 stacked images or so
DOWNSIZE = numpy.array([335, 253])


# TODO: Try different distances: two best candidates are euclidean and
# cosine. Pass into NearestNeighbors as 'metric'
def inspect_knn(embeddings, impaths, sample, k, imdir, savedir):

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embeddings)

    samples = numpy.random.choice(range(len(impaths)), size=sample, replace=False)
    # Whenever we use the distances or indices we must skip the first one, that
    # is the original point
    sample_dists, sample_indices = nn.kneighbors(embeddings[samples])

    for i, (index, k_dists, k_indices) in enumerate(
        zip(samples, sample_dists, sample_indices)
    ):
        highlight_images(
            hero=impaths[index],
            others=[impaths[k_index] for k_index in k_indices[1:]],
            path=savedir.joinpath(f"knn_sample_{i:02}.jpg"),
            size=DOWNSIZE,
        )
    print(f"Saved {sample} {k}-nn images to {savedir}")


def inspect_differentials(embeddings, impaths, sample, k, savedir):
    """
    Sample randomly X times and then show oher random samples but sorted so
    that the viewer can see the respective distances
    """

    samples = numpy.random.choice(range(len(impaths)), size=sample, replace=False)

    # TODO: Build in cosine distance capability
    for i, index in enumerate(samples):

        # Check a feasibly large number of vectors, select the largest distance
        # and also the closest to 0.5
        subsamples = numpy.random.choice(range(len(impaths)), size=k, replace=False)
        distances = numpy.linalg.norm(embeddings[subsamples] - embeddings[index], axis=1)

        # Get the max and the closest to 50%
        maxind = distances.argmax()
        maxval = distances.max()
        midind = abs((distances / maxval) - 0.5).argmin()
        midval = distances[midind]
        lowind = abs((distances / maxval) - 0.2).argmin()
        lowval = distances[lowind]

        highlight_images(
            hero=impaths[index],
            others=[impaths[subsamples[ind]] for ind in (lowind, midind, maxind)],
            path=savedir.joinpath(f"differential_sample_{i:02}.jpg"),
            size=DOWNSIZE,
            pad=20,
            text=[f"{lowval:.3}", f"{midval:.3}", f"{maxval:.3}"],
        )

        # Also save a histogram of the distance distributions
        pyplot.hist(x=distances)
        pyplot.xlabel("Distance from sample")
        pyplot.ylabel("Count")
        pyplot.title(f"Histogram for distance from {impaths[index].name}")
        pyplot.tight_layout()
        pyplot.savefig(savedir.joinpath(f"differential_sample_{i:02}_hist.jpg"))
        pyplot.close()

    print(f"Saved {sample} differential images to {savedir}")


def visualize_2d(embeddings, metapaths, colorfield, savedir):

    # Apparently TSNE is better than PCA for displaying neighbors correctly,
    # but warps the global space. We want to see if the color field is
    # captured by the neighbor clusters, so let's use TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embed_2d = tsne.fit_transform(embeddings)

    # Get the colors from the metadata
    colors = [json.load(mp.open("r"))[colorfield] for mp in metapaths]

    pyplot.figure(figsize=(10, 8))
    scatter = pyplot.scatter(embed_2d[:, 0], embed_2d[:, 1], c=colors, cmap="RdYlGn")
    pyplot.colorbar(scatter, label=colorfield)
    pyplot.xlabel("TSNE projection 1")
    pyplot.ylabel("TSNE projection 2")
    pyplot.title(f"Project embeddings from {embeddings.shape[1]}-d to 2-d, see clusters")
    pyplot.tight_layout()
    pyplot.savefig(savedir.joinpath(f"embeddings_in_2d.jpg"))
    pyplot.close()

    print(f"Saved embeddings in 2d image to {savedir}")


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
        help="Directory with all images we want to examine and corresponding"
        " json metadata",
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
    impaths = [args.image_directory.joinpath(im) for im in imnames]
    metapaths = [p.with_suffix(".json") for p in impaths]

    inspect_knn(
        embeddings,
        impaths,
        sample=10,
        k=4,
        imdir=args.image_directory,
        savedir=args.save_directory,
    )

    inspect_differentials(
        embeddings,
        impaths,
        sample=10,
        k=2000,
        savedir=args.save_directory,
    )

    visualize_2d(
        embeddings,
        metapaths,
        colorfield="harvestability_label",
        savedir=args.save_directory,
    )