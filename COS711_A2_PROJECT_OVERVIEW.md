# COS711 Assignment 2: Solar Filament Segmentation

**Assignment deadline:** 1 November 2026 at 23:30\
**Project type:** Deep-learning computer vision\
**Primary evaluation metric:** Panoptic Quality (PQ)

## 1. Project overview

This project investigates how deep-learning methods can automatically
identify and segment solar filaments in H-alpha images of the Sun. Solar
filaments are elongated structures of relatively cool, dense plasma
suspended in the solar atmosphere. In H-alpha observations, they can
appear as dark, thread-like or curved features against the brighter
solar disk. Their shapes can be thin, irregular, fragmented, and
sometimes close to or touching other filaments, which makes accurate
segmentation challenging.

The aim is to develop a model that identifies the pixels belonging to
each filament and, importantly, distinguishes one filament from another.
The project is therefore not only about detecting whether a pixel
belongs to a filament; it is also about recovering individual filament
instances with shapes that agree with the annotations.

The work uses the MAGFiLO 1.0 dataset supplied for the Kaggle
competition. The labelled training data is used to develop the model and
evaluate it locally. The competition's unlabelled test images are used
only to generate predictions for the final submission. Test labels from
other releases or external sources must not be obtained or used.

## 2. The dataset

The dataset folder supplied for the assignment has the following general
layout:

``` text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

The image files are grayscale JPEG observations in the H-alpha spectral
line. The supplied images are described as 2048 × 2048 pixels. The
training directory contains the images and a COCO-style JSON file
containing their annotations. The test directory contains images for
which the competition dataset does not supply ground-truth annotations.

The dataset archive supplied for this project contains approximately 888
files and is about 751 MB in size. File counts can vary if the archive
or directory includes additional metadata, so scripts should inspect the
extracted contents rather than rely solely on the archive count.

### 2.1 What the images look like

A typical input is a square, grayscale solar image. The solar disk and
its intensity patterns form the background, while filaments appear as
dark, narrow, elongated structures. Some filaments are long and curved;
others are shorter, faint, irregular, or close to neighbouring
structures. Background texture, intensity variation, and the fine width
of the filaments make the boundaries difficult to identify consistently.

A segmentation result should overlay the filament regions on the
original image. For instance segmentation, each separate filament should
have its own instance identity rather than all filaments being combined
into one foreground region.

The project should inspect real examples from the supplied dataset
before selecting preprocessing or model settings. This helps reveal the
range of filament shapes, the degree of contrast variation, touching
structures, faint filaments, and annotation differences.

## 3. Ground-truth annotations

The training labels are stored in:

`MAGFiLO_1.0_Annotations_kaggle2026_train.json`

This is a COCO-style annotation file rather than a directory of
ready-made raster masks. It contains image records, annotation records,
and category definitions.

### 3.1 Image records

Each entry in the JSON `images` array describes an annotated image
record. Relevant fields include:

-   `id`: the identifier for that annotation record.
-   `file_name`: the image filename.
-   `width` and `height`: the image dimensions.

An important feature of this dataset is that the same physical solar
observation can appear in multiple annotation records, for example when
different annotators have labelled the same filename. These records may
have different IDs and different annotations. They should not be treated
as independent physical observations when creating the training and
validation split.

### 3.2 Filament annotations

Each entry in the `annotations` array describes an annotated filament
instance. Fields include:

-   `image_id`: the image record to which the annotation belongs.
-   `category_id`: the annotation category.
-   `segmentation`: polygon coordinates outlining the filament.
-   `area`: the annotated area.
-   `bbox`: the bounding box.
-   `iscrowd`: the COCO-style crowd flag.

The polygon coordinates provide the ground truth from which raster masks
can be generated programmatically. There is no need to draw the training
masks manually. For a conventional segmentation model, the polygons can
be rasterised into a binary foreground mask. For instance-aware training
and evaluation, each filament should retain a distinct instance ID, and
its category information should be preserved separately where required.

A single integer-valued instance mask can represent only one instance ID
at a pixel. If annotated polygons overlap, the conversion process must
define how that overlap is handled and should preserve the original
polygons or separate per-instance masks so information is not silently
lost.

### 3.3 Annotation categories

The supplied JSON defines these categories:

    Category ID Category name
  ------------- ----------------
              1 Left
              2 Right
              3 Unidentifiable
              4 Ambiguous

These names should be retained as supplied until the competition
documentation clarifies exactly how categories affect the official
evaluation and submission. The inspected annotation file contained no
annotations with category ID 4, but the code should still handle that
category safely if it appears in another valid file or dataset revision.

Different annotators may disagree about the boundary, extent, or
presence of a filament. The project should therefore preserve each
annotation record and avoid automatically merging annotations from
different annotators unless the official competition protocol explicitly
requires it.

## 4. Dataset inspection and preparation

Before training, the dataset needs to be checked for missing files,
duplicate filenames, corrupt images, inconsistent dimensions, invalid
polygon coordinates, and image records without annotations. Image
dimensions in the JSON should be compared with the actual image
dimensions.

The data preparation stage will create a reproducible training and
validation split. Related annotation records for the same physical
observation must stay in the same split. Where the available metadata
supports it, the grouping strategy should also consider observations
that are close in time or otherwise closely related, since
near-duplicate observations can make validation results overly
optimistic.

The split manifests should identify each JSON image record by its
`image_id` and retain its filename. This is important because multiple
annotation records may refer to the same physical image but carry
different labels. The original images and JSON should remain unchanged;
generated files should be stored separately.

After splitting, the polygon annotations can be converted into masks.
The image and mask must undergo matching geometric transformations
during preprocessing and augmentation. Image interpolation and mask
interpolation must be chosen appropriately: masks should not acquire
artificial fractional labels during resizing. Augmentations should be
applied to training data only and should preserve plausible
solar-filament morphology.

## 5. Segmentation problem and model outputs

The model receives a grayscale solar image and predicts the location and
shape of solar filaments.

There are two related representations:

**Semantic segmentation** labels each pixel as background or filament.
This is useful as a baseline, but a single binary mask does not identify
which filament each foreground pixel belongs to.

**Instance segmentation** assigns a separate identity to each filament.
This is more directly aligned with an instance-level metric such as
Panoptic Quality. If a model predicts only a semantic mask, the project
must define and evaluate an instance-recovery method, such as connected
components or a justified separation procedure. These methods can fail
when neighbouring filaments touch or one filament is broken into
multiple pieces, so they must be evaluated rather than assumed to work.

The project may compare a straightforward segmentation baseline with
stronger architectures or instance-aware methods. Model choice should be
guided by validation performance, qualitative inspection, computational
cost, and compatibility with the official submission format.

## 6. Evaluation: Panoptic Quality

The primary metric for the assignment is Panoptic Quality (PQ). PQ
evaluates instance segmentation by considering both the quality of the
matched segment shapes and whether instances are correctly recognised.
It is therefore sensitive to errors such as:

-   missing a real filament;
-   predicting a filament where none is annotated;
-   splitting one annotated filament into several predicted instances;
-   merging multiple annotated filaments into one predicted instance;
-   producing inaccurate filament boundaries.

Pixel-level measures such as Dice score, intersection over union,
precision, and recall can still be useful diagnostic metrics, but they
do not replace local PQ validation.

The local implementation of PQ must follow the official Kaggle
competition's matching threshold, instance-matching rules,
ignored-region handling, category treatment, and aggregation procedure.
These details should be verified from the competition documentation
rather than guessed. Unit tests should cover perfect matches, missing
instances, false positives, empty masks, split instances, and merged
instances.

## 7. Experiments and improvement strategy

The project should establish a reproducible baseline and then compare
controlled changes. Possible experimental dimensions include
architecture, input resolution, loss function, augmentation,
thresholding, instance extraction, and post-processing.

The main objective is to improve segmentation quality without
sacrificing the identity and continuity of individual filaments. The
analysis should pay particular attention to:

-   **Completeness:** recovering entire filaments, including thin and
    faint sections.
-   **False positives:** avoiding background structures being labelled
    as filaments.
-   **Continuity:** reducing breaks and isolated fragments within a
    filament.
-   **Instance separation:** distinguishing neighbouring or touching
    filaments.
-   **Boundary quality:** matching the annotated filament outlines.
-   **Efficiency:** keeping model size, training duration, and inference
    time practical.

Post-processing methods such as connected components or watershed may be
tested when appropriate. Thresholds, minimum instance sizes, and
morphological operations must be selected using the validation set, not
the unlabelled test images. Each change should be compared against the
same baseline using local PQ and documented timing measurements.

Where feasible, experiments should use multiple random seeds and report
the mean and standard deviation. Visual examples of both strong and weak
predictions are needed to explain the results, not just a single metric
score.

## 8. Efficiency and model compression

The assignment also considers computational efficiency. The final report
should record the model's parameter count and file size, training time,
and end-to-end inference time. Timing should state the hardware used and
whether it includes image loading, preprocessing, model prediction,
instance extraction, and post-processing.

Compression techniques such as pruning, distillation, a smaller
architecture, or reduced precision may be explored where feasible. A
compressed model should be evaluated with the same validation process as
the original model so the trade-off between PQ, size, and speed is
measured fairly.

## 9. Final prediction and submission

After selecting a final approach using the labelled training data and
held-out validation results, the chosen model is used to predict the
unlabelled competition test images. The predictions must be serialized
into `submission.csv` using the exact schema and encoding required by
the Kaggle competition.

The submission-generation code should verify the required columns,
identifiers, row count, duplicate rows, and prediction serialization.
The test images must not be used as labelled training examples, and no
external test annotations may be obtained or used.

The final deliverable is a ZIP containing the project code, the exact
`submission.csv` uploaded to Kaggle, the PowerPoint slides, and the
presentation video. The video must be no longer than 10 minutes and no
larger than 50 MB, and it must feature every team member and describe
their responsibilities. The team must contain two or three students;
single-student submissions are not accepted according to the assignment
brief.

## 10. Suggested codebase organisation

A maintainable project structure separates data handling, models,
metrics, training, inference, and evaluation:

``` text
COS711_A2/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   └── experiments.yaml
├── docs/
│   └── competition_spec.md
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── data/
│   ├── raw/
│   │   └── MAGFiLO_1.0_Kaggle_2026/
│   ├── splits/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── dataset.py
│   │   └── split_data.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── losses.py
│   ├── metrics/
│   │   ├── panoptic_quality.py
│   │   └── pixel_metrics.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── predict.py
│   │   ├── postprocess.py
│   │   └── make_submission.py
│   └── utils/
│       ├── reproducibility.py
│       └── timing.py
├── tests/
│   ├── test_data.py
│   ├── test_panoptic_quality.py
│   └── test_submission.py
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
├── presentation/
│   ├── slides.pptx
│   └── video.mp4
└── submission.csv
```

The raw dataset should not be edited in place. Generated splits, raster
masks, predictions, checkpoints, metrics, and figures belong in their
corresponding output directories. Large data and model files should not
be committed to version control by default.

## 11. Project outcome

The intended outcome is a reproducible deep-learning system that
segments solar filaments in H-alpha images, preserves individual
filament instances as required by the competition, and produces a valid
Kaggle submission. The project report and presentation should explain
the dataset, the annotation-to-mask process, the leakage-safe validation
strategy, the models and experiments, local PQ results, observed failure
modes, and the measured trade-off between accuracy and computational
efficiency.

The work should distinguish verified competition requirements from
implementation choices. In particular, the exact submission format and
the detailed PQ rules must be checked against the official Kaggle
Overview, Data, and Rules pages before final implementation.

## References

1.  A. Ahmadzadeh et al., "MAGFiLO: Manually Annotated GONG Filaments
    from H-Alpha Observations," *Scientific Data*, 11:1031, 2024.
    https://doi.org/10.1038/s41597-024-03876-y
2.  A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár, "Panoptic
    segmentation," *Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR)*, 2019.
    https://doi.org/10.1109/CVPR.2019.00963
