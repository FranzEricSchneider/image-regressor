import argparse
from collections import defaultdict
import cv2
import json
from matplotlib import pyplot
import numpy
from pathlib import Path
from shutil import copy

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from image_regressor.vis import highlight_images


# Size that is good for downsampled viewing of many images, on the order of
# 3x5 stacked images or so
DOWNSIZE = numpy.array([335, 253])


# TODO: Rework this to handle existing assignments
def assign(embed_file, chart, imdir, savedir, subfolders):
    """
    Use embeddings to assign different reviewers to as diverse a set of samples
    as possible.

    Arguments:
        embed_file: pathlib.Path object that contains embeddings in a json dict
            as seen elsewhere in this file
        chart: List of dictionaries containing
            {
                "id": <user id/name>,
                "number": <number of images to assign>,
                "crossover": <OPTIONAL dictionary of {"id": fraction} where we
                    can have X% of user B cross over with user A>
            }
        imdir: pathlib.Path object to a directory with all the images in the
            embedding data
        savedir: pathlib.Path object to the directory we want to save the
            assignment folders in
        subfolders: Iterable of directory names that we want to make inside of
            the assignment folder
    """

    # Load embeddings
    data = json.load(embed_file.open("r"))["data"]
    imnames = sorted(data.keys())
    embeddings = numpy.array([data[key] for key in imnames])

    assignments = defaultdict(list)

    def unassigned(uid=None):
        """
        Helper function to get all currently unassigned image indices OR just
        the indices not assigned to the given ID
        """
        total = set(range(len(imnames)))
        if uid is None:
            for assigned in assignments.values():
                total -= set(assigned)
        else:
            total -= set(assignments[uid])
        return sorted(total)

    # Assign images to users in the order that that are given in the chart
    for user in chart:
        uid, N = user["id"], user["number"]

        # If they have crossover images, sample a crossover amount from images
        # already assigned to other users
        if "crossover" in user:
            for cross_id, f_N in user["crossover"].items():
                cross_N = int(N * f_N)
                assignments[uid].extend(
                    sample(
                        embeddings,
                        pool=assignments[cross_id],
                        n_required=cross_N,
                    )
                )

        # Then fill in the remaining images
        remaining = N - len(assignments[uid])
        if remaining > 0:
            # If the UNASSIGNED IMAGES do not have enough images left to fill
            # the user's quota, start drawing from the backup pool which is any
            # image not already assigned to this user (will result in more
            # overlap)
            assignments[uid].extend(
                sample(
                    embeddings,
                    pool=unassigned(),
                    backup_pool=unassigned(uid),
                    n_required=remaining,
                )
            )

        # Save the images and desired subfolders to a user folder
        userdir = savedir.joinpath(uid)
        userdir.mkdir(parents=True, exist_ok=False)
        for subfolder in subfolders:
            userdir.joinpath(subfolder).mkdir()
        if len(assignments[uid]) != N:
            import ipdb

            ipdb.set_trace()
        for imname in map(lambda x: imnames[x], assignments[uid]):
            copy(imdir.joinpath(imname), userdir.joinpath(imname))


def sample(embeddings, pool, n_required, backup_pool=None, f_kmeans=0.7):
    """
    Sample a set of embedding indices from the pool (and if necessary, the)
    backup pool.

    Right now we do this in two stages:
        A kmeans stage, where we find k clusters in the data and take an image
            index from each cluster. The point is to spread the samples out in
            the embedding space
        A knn stage, where we sample indices whose nearest neighbor in the
            current sample set is the furthest away. The point is to get
            outliers that may not form nice clusters.
    The fraction of how many should be taken via kmeans vs knn is tunable, but
    I'm not sure it matters much.

    If the number required is greater than the current pool size, AND a backup
    pool has been provided, then first take all of the pool candidates and then
    sample from the backup pool.

    Arguments:
        embeddings: MxN array of image embeddings, for M images with a feature
            length of N
        pool: list of indices into embeddings that are available for sampling
        n_required: number of indices to sample
        backup_pool: (optional) more expansive set of indices in case
            n_required is larger than the pool. If it is None we simply return
            everything in pool. If not None and we overflow then sample from
            the backup pool
        f_kmeans: the fraction of samples to take using kmeans rather than
            furthest neighbor sampling
    """

    indices = []
    if n_required > len(pool):
        indices.extend(pool)
        pool = sorted(set(backup_pool) - set(indices))
        n_required -= len(indices)

    n_kmeans = int(n_required * f_kmeans)
    n_knn = n_required - n_kmeans

    # Step 1: KMeans clustering
    indices.extend(kmeans_sample(pool, embeddings[pool], n_kmeans))

    # Take the initial selection out of the pool
    pool = sorted(set(pool) - set(indices))

    # Step 2: Use knn to get the outliers
    indices.extend(
        knn_sample(
            in_indices=indices,
            in_embed=embeddings[indices],
            out_indices=pool,
            out_embed=embeddings[pool],
            k=n_knn,
        )
    )

    return indices


def kmeans_sample(indices, embeddings, k):
    """Take a random sample index from each of the k clusters."""

    kmeans = KMeans(n_clusters=k, n_init="auto")
    clusters = kmeans.fit_predict(embeddings)

    # Iterate through clusters and sample one index from each cluster
    indices = numpy.array(indices)
    sampled_indices = []
    for cluster_idx in range(k):
        sampled_indices.append(numpy.random.choice(indices[clusters == cluster_idx]))

    return sampled_indices


def knn_sample(in_indices, in_embed, out_indices, out_embed, k):
    """
    Find the k indices that are furthest from the existing assignments (in) in
    the available embeddings (out)
    """

    knn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    knn.fit(in_embed)

    dists, _ = knn.kneighbors(out_embed)
    furthest = numpy.argsort(dists.squeeze())[::-1]

    return numpy.array(out_indices)[furthest[:k]].tolist()


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
        distances = numpy.linalg.norm(
            embeddings[subsamples] - embeddings[index], axis=1
        )

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
    pyplot.title(
        f"Project embeddings from {embeddings.shape[1]}-d to 2-d, see clusters"
    )
    pyplot.tight_layout()
    pyplot.savefig(savedir.joinpath(f"embeddings_in_2d.jpg"))
    pyplot.close()

    print(f"Saved embeddings in 2d image to {savedir}")


def visualize_other(embed_dict_0, embed_dict_1, savedir):
    deltas = numpy.array(
        [
            (numpy.array(embed_dict_0[key]) - numpy.array(embed_dict_1[key]))
            for key in embed_dict_0
            if key in embed_dict_1
        ]
    )
    distances = numpy.linalg.norm(deltas, axis=1)

    pyplot.hist(x=distances)
    pyplot.xlabel("Distance between paired images")
    pyplot.ylabel("Count")
    pyplot.title(f"Histogram of pair distances, mean {distances.mean():.4}")
    pyplot.tight_layout()
    pyplot.savefig(savedir.joinpath(f"other_embeddings_hist.jpg"))
    pyplot.close()

    print(f"Saved pairwise comparisn to other embeddings {savedir}")


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
    parser.add_argument(
        "-o",
        "--other-embeddings",
        help="Compare against other embeddings (only useful if from the same"
        " model, and of the same images)",
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

    if args.other_embeddings is not None:
        other_data = json.load(args.other_embeddings.open("r"))["data"]

        visualize_other(
            data,
            other_data,
            savedir=args.save_directory,
        )
