# Changelog

## [Unreleased]

### Fixed
- Refuse missing/unreadable segmentation masks instead of manufacturing zero labels.
- Categorical nearest-neighbor mask resizing; removed unverified historical augmentation and credential/download cells.

### Added
- Strict image/mask/patient manifest, patient-disjoint split, binary-mask validation and explicit Dice convention.
- Typed CPU U-Net research CLI, actual synthetic backend tests, uv lock, quality checks and data-only Docker/CI.
- Honest clinical/data limitations. No trained MRI quality claim, license change or main merge.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.
