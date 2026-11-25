# Ordinary image-mask inference, 25 November 2026

`medical-segmentation predict IMAGE.png MODEL/model.keras NEW_MASK.png` loads an explicit trusted local grayscale square Keras U-Net artifact. Check installed `--help` for the entrypoint. It applies the package training grayscale resize/divide255 preprocessing, thresholds probabilities at0.5 and restores original pixel dimensions with nearest-neighbor interpolation. Existing output paths are refused. Invalid/nonfinite/out-of-range model outputs fail.

This does not make nuclei weights into an MRI model. Only use a matching-modality artifact and preprocessing. Current BBBC039 evidence is nuclei microscopy, not brain MRI. Original MRI source masks/weights are not newly recovered. Research output is not diagnosis. Synthetic tests validate dimensions, binary pixels, CLI and no overwrite, not tumor accuracy.
