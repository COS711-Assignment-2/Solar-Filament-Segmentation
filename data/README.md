# Data directories

Place the KaggleHub download or an extracted competition archive under
`data/raw/`. The preparation pipeline discovers the actual enclosing directory
and does not modify raw files. Generated split manifests belong in
`data/splits/`; generated masks and preprocessed images belong in
`data/processed/`.

The test set is unlabelled and must not be used to fit preprocessing
statistics, select hyperparameters, or evaluate local ground truth.
