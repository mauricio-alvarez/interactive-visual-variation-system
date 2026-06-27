# Data Directory

Use `data/raw` for downloaded or collected datasets and `data/processed` for resized/filtered images.

The portrait fine-tuning workflow uses this ignored local layout:

```text
data/portrait_finetune/
  raw/
    consented/
    licensed/
  metadata/
  processed/
    train/
    validation/
    test/
```

Run `scripts/prepare_portrait_dataset.py` to create processed splits and metadata from consented or explicitly licensed portraits.

For a non-commercial FFHQ research run, clone `NVlabs/ffhq-dataset` and use `scripts/download_ffhq_subset.py` to fetch a bounded 1024x1024 subset before preparing the splits.

Do not commit raw datasets unless their license and size are appropriate for GitHub.
