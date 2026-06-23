# Project path
import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ['PROJECT_ROOT']
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from nets.unet import UNetCondition2D
from dataset.fundus import FundusDataLoader
from diffusion.diffusion_classifier import DiffusionClassifier
from utils.metrics import Accuracy, F1, Precision, Recall, AUC
from utils.wavelet import wavelet_enc_2

# Third party imports
import torch
from safetensors.torch import load_file
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import AutoencoderKL
import accelerate

# Training configuration
class TrainingConfig:
    def __init__(self):
        config_str = os.environ.get('CONFIG')
        if config_str is None:
            raise ValueError("CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config['project_root']
        self.experiment_dir = self.config['experiment_dir']

        # Construct experiment path
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)


def load_pretrained_weights(diffusion_classifier, unet, checkpoint_dir):
    """
    Load pretrained model weights (UNet + EMA + class encoder) from a checkpoint
    directory produced by `accelerator.save_state` for finetuning.

    Only the network weights are loaded. The optimizer, lr-scheduler, gradient
    scaler, RNG state and epoch counter are intentionally NOT restored, so that
    finetuning starts a fresh optimization run (fresh momentum, new learning
    rate, epoch 0) on top of the pretrained weights.

    Expected files (accelerate naming convention):
        model.safetensors    -> self.model (UNet)
        model_1.safetensors  -> self.ema   (EMA wrapper)
        model_2.safetensors  -> self.encoder (nn.Embedding of the classes)
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"pretrained_checkpoint directory not found: {checkpoint_dir}"
        )

    # 1) UNet backbone (required)
    unet_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"Expected UNet weights at {unet_path}")
    unet.load_state_dict(load_file(unet_path))
    print(f"Loaded UNet weights from {unet_path}")

    # 2) EMA weights (required - inference/classification uses the EMA model)
    ema_path = os.path.join(checkpoint_dir, "model_1.safetensors")
    if not os.path.exists(ema_path):
        raise FileNotFoundError(f"Expected EMA weights at {ema_path}")
    diffusion_classifier.ema.load_state_dict(load_file(ema_path))
    print(f"Loaded EMA weights from {ema_path}")

    # 3) Class-conditioning encoder (nn.Embedding). Same shape for any binary
    #    task (classes+1 = 3 embeddings), so it loads as a warm start and is
    #    refined during finetuning.
    encoder_path = os.path.join(checkpoint_dir, "model_2.safetensors")
    if diffusion_classifier.encoder is not None and os.path.exists(encoder_path):
        diffusion_classifier.encoder.load_state_dict(load_file(encoder_path))
        print(f"Loaded class encoder weights from {encoder_path}")
    else:
        print("No class encoder weights loaded (starting from init)")


def fundus_plotter(output_dir: str, batches: list, samples: list, epoch: int, process_idx: int):
    """
    Plot Fundus samples and save them to the output_dir

    output_dir: str
        The output directory to save the plots
    batches: list
        List of batches of images
    samples: list
        List of samples
    epoch: int
        The epoch number of the training
    process_idx: int
        The process index

    Returns
        image_path: str
            The path to the saved image
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    for i, (batch, sample) in enumerate(zip(batches, samples)):
        images = batch["images"]
        prompts = batch["prompt"]
        samples = sample

        for j in range(1): # batch size

            if config.wavelet_transform:
                sample_item = samples[j] * 2 # [-2, 2]
                sample_item = wavelet_enc_2(sample_item) # [-1, 1]

                image = images[j] * 2 # [-2, 2]
                image = wavelet_enc_2(image) # [-1, 1]
            else:
                sample_item = samples[j]
                image = images[j]

            pred = sample_item.cpu().detach().numpy() / 2 + 0.5 # [-1, 1] -> [0, 1]
            image = image.cpu().detach().numpy() / 2 + 0.5 # [-1, 1] -> [0, 1]

            prompt = prompts[j]
            activity = "active" if prompt else "inactive"

            fig, axs = plt.subplots(1, 2, figsize=(5, 5))

            axs[0].imshow(pred.transpose(1, 2, 0))
            axs[0].axis('off')
            axs[0].set_title("Sample Prediction")
            axs[1].imshow(image.transpose(1, 2, 0))
            axs[1].axis('off')
            axs[1].set_title("Sample Image")

            # Set top row title
            fig.suptitle(f"Patient status: {activity}", fontsize=16)
            plt.tight_layout()

            # Make path for patient
            patient_path = os.path.join(output_dir, f"{activity}")
            os.makedirs(patient_path, exist_ok=True)
            image_path = os.path.join(patient_path, f"epoch_{epoch}_sample_{j}_process_{process_idx}.png")
            plt.savefig(image_path, dpi=300)
            plt.close()

    return image_path

def main():
    global config
    config = TrainingConfig()

    if config.resume and config.pretrained_checkpoint:
        raise ValueError(
            "RESUME=1 and pretrained_checkpoint cannot both be set. "
            "To continue finetuning from an existing fundus checkpoint, set RESUME=1 and PRETRAINED_CHECKPOINT=\"\"."
        )
    if not config.resume and not config.pretrained_checkpoint:
        raise ValueError(
            "finetune.py requires either RESUME=1 (continue existing run) or "
            "pretrained_checkpoint (start finetuning from a new base). "
            "Use train.py for training from scratch."
        )

    # Set seed
    accelerate.utils.set_seed(config.seed)

    data = FundusDataLoader(
        data_path=config.data_path,
        wavelet_transform=config.wavelet_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    test_loader = data.get_test_loader()

    # Define model - identical architecture to the pretrained ISIC UNet
    unet = UNetCondition2D(
        sample_size=config.image_size if not config.wavelet_transform else config.image_size//2,
        in_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        out_channels=config.image_channels if not config.wavelet_transform else 4*config.image_channels,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 512, 1024),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        encoder_hid_dim=512,
        encoder_hid_dim_type='text_proj',
        cross_attention_dim=512,
    )

    # Create the diffusion classifier object
    diffusion_classifier = DiffusionClassifier(
        backbone=unet,
        config=config,
    )

    # Load the pretrained ISIC weights into UNet + EMA + class encoder (skipped when resuming).
    # This happens BEFORE accelerator.prepare() (called inside train_loop), so
    # the wrapped modules carry the pretrained weights. Optimizer/scheduler are
    # built fresh below -> a clean finetuning run, not a resume.
    if config.pretrained_checkpoint:
        load_pretrained_weights(diffusion_classifier, unet, config.pretrained_checkpoint)

    # Define optimizer and scheduler (fresh, with the finetuning learning rate)
    params = list(unet.parameters())

    if diffusion_classifier.encoder is not None:
        params += list(diffusion_classifier.encoder.parameters())

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    metrics = [Accuracy("accuracy"), F1("f1"), Precision("precision"), Recall("recall"), AUC("auc")]

    # Finetune the model. New checkpoints are written to
    # experiments/fundus-unet/checkpoints and best_checkpoint as usual.
    diffusion_classifier.train_loop(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        checkpoint_metric="f1",
        plot_function=fundus_plotter
    )

if __name__ == "__main__":
    main()
