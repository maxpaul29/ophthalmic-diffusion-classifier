import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ["PROJECT_ROOT"]
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from nets.resnet import ResNet2D
from nets.efficient import EfficientNet2D
from nets.vit import ViT2D
from nets.swin import Swin2D

from dataset.fundus import FundusDataLoader
from classifier.classifier import Classifier
from utils.metrics import Accuracy, F1, Precision, Recall, AUC

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate


class InferenceConfig:
    def __init__(self):
        config_str = os.environ.get("CONFIG")
        if config_str is None:
            raise ValueError("CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config["project_root"]
        self.experiment_dir = self.config["experiment_dir"]
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)


def main():
    global config
    config = InferenceConfig()

    # Set random seed
    accelerate.utils.set_seed(config.seed)

    fundus = FundusDataLoader(
        wavelet_transform=False,
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=(config.image_size, config.image_size),
        split_prefix=config.split_prefix or "fundus"
    )
    
    train_loader = fundus.get_train_loader()
    val_loader = fundus.get_val_loader()
    test_loader = fundus.get_test_loader()

    if config.backbone == "resnet":
        backbone = ResNet2D(
            variant=config.variant,
            pretrained=config.pretrained,
            in_channels=config.image_channels,
        )
    elif config.backbone == "efficientnet":
        backbone = EfficientNet2D(
            variant=config.variant,
            pretrained=config.pretrained,
            in_channels=config.image_channels,
        )
    elif config.backbone == "vit":
        backbone = ViT2D(
            variant=config.variant,
            pretrained=config.pretrained,
            in_channels=config.image_channels,
            img_size=(config.image_size, config.image_size)
        )
    elif config.backbone == "swin":
        backbone = Swin2D(
            variant=config.variant,
            pretrained=config.pretrained,
            in_channels=config.image_channels,
            img_size=(config.image_size, config.image_size)
        )
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")
    
    classifier = Classifier(
        backbone=backbone,
        config=config
    )

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )

    metrics = [
        Accuracy("accuracy"),
        F1("f1"),
        Precision("precision"),
        Recall("recall"),
        AUC("auc")
    ]


    val_loss, metric_output = classifier.inference(
        optimizer=optimizer,
        train_dataloader=train_loader,  # might not be used but provided for consistency
        val_dataloader=test_loader,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        checkpoint_folder=config.checkpoint_folder,
    )

    results = [{k: round(v.item(), 4) for k, v in d.items()} for d in metric_output]
    print("Inference Results:", results)

    # Also write the results as JSON next to the checkpoint used for this run,
    # so a cross-validation orchestrator (aggregate_cv_results.py) can read
    # each fold's metrics back in without parsing console output.
    merged = {}
    for d in results:
        merged.update(d)
    result_path = os.path.join(config.checkpoint_folder, "inference_result.json")
    with open(result_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()