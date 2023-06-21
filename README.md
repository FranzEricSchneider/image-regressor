# image-regressor
Simple repo for doing image regression using PyTorch

Note: some visualization code was adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations [MIT License]

***

### How to start training
1. Put all of your data into `<directory>/train/` and `<directory>/test`. You should have a series of images, and each image should have a corresponding `.json` file of the same name (with the suffix changed) where the variable you're trying to regress is in each `.json` file. **[See example code]**
2. Choose your configuration settings in `config.py`. Note that some of these are available as command-line arguments (seen in `util.py`) for sweep purposes. There are many config options, but some initially important ones are
    - `extension` [string] Image suffix to search for, e.g. "jpg" or "png".
    - `regression_key` [string] This is the name of the field in the `.json` files we want to regress.
    - `wandb` [boolean] If True, this attempts to log into [wandb.ai](https://wandb.ai/) and record the results. If you have not set this up, set to False.
    - `use_existing` [string] You can choose between pre-existing pytorch CNN encoders. If `None` is chosen, then a simple CNN encoder (governed by the `cnn_*` config variables) is built.
    - `output_limit` [integer / float] If `None` is given, the output is unconstrained. If a number is given, the output is passed through a sigmoid and scaled between 0 and `output_limit`.
3. Choose your augmentations (look at `EXAMPLE_augmentations.json` for possibilities, and point to your new/modified augmentation files in `config.py`
    - In order to find the mean/std for your dataset to populate the train/test augmentation files, run `python loader.py --image-directory <directory>/train/`
4. Run the training code **[See example code for automation options]**
    - `python main.py <directory>`

#### Example Code
**HERE** is example code for taking image paths and `.json` paths, then splitting that up into train/test directories.
##### Arguments
- `impaths` input is a list of all image paths, where the `.json` files are assumed to be in the same directory as each image
- `directory` is where you want the split images to end up
- `train_size` is the fraction of the data you want to end up as train (vs. test)
- `downsample` is the integer amount you want to downsample the images, if any
```
def nn_data(impaths, directory, train_size=0.8, downsample=4):
    directory.joinpath("train").mkdir(parents=True, exist_ok=False)
    directory.joinpath("test").mkdir(parents=True, exist_ok=False)
    train_ims, test_ims = train_test_split(impaths,
                                           train_size=train_size,
                                           random_state=42)
    # Resize and save
    for split, split_images in [("train", train_ims),
                                ("test", test_ims)]:
        for impath in split_images:
            new_path = directory.joinpath(split, impath.name)
            print(impath, " -> ", new_path)
            image = Image.open(impath)
            if downsample > 1:
                new_size = (image.size[0] // downsample,
                            image.size[1] // downsample)
                downsampled = image.resize(new_size, resample=Image.Resampling.LANCZOS)
                downsampled.save(new_path)
            else:
                image.save(new_path)
            copy(impath.with_suffix(".json"), new_path.with_suffix(".json"))

    glob_str = f"*{impaths[0].suffix}"
    for split in ("train", "test"):
        subdir = directory.joinpath(split)
        number = len(sorted(subdir.glob(glob_str)))
        print(f"{number} image files in {subdir}")
```

**HERE** is example code for running a sweep of networks and augmentation files. See `utils.py / load_config()` for what these arguments do:
```
import subprocess

arguments = [
    "--use-existing mobilenet_v3_small",
    "-D 4  -W 64   -R 0.10 -A  --use-existing mobilenet_v3_small",
    "-D 6  -W 64   -R 0.14 -A  --use-existing efficientnet_b0",
]
augments = ["train_aug0.json", "train_aug1.json", "train_aug2.json"]

for arg in arguments:
    for aug in augments:
        # Construct the bash command
        command = [
            "python",
            "main.py",
            "/hdd/nn_data/regressor/pw_data_div4/",
            "-g", aug
        ] + arg.split()
        # Run the bash command and wait for it to finish
        subprocess.run(command, check=True)
```

***

### Evaluation

#### Visualize your labels

In order to visualize your labels, you can run

```
python3 vis.py --source from_file --regression-key <key> --image-directory <directory> --output-directory <out-dir>
```

Based on the arguments, you can optionally create video output, as well as sorting based on `.json` fields [TODO].

#### Visualize model results

In order to visualize your model results, you can run

```
python3 vis.py --source from_model --regression-key <key> --image-directory <directory> --output-directory <out-dir> --wandb-run-path <run path> --wandb-keyfile wandb.json
```

This has the same options as the label visualization, but will also load the model given and run it on each image.