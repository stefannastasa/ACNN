import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from timm import create_model
import wandb
from traininig_pipeline import TrainingPipeline
from models import PreActResNet18, MLP, LeNet


class PipelineWrapper:
    def __init__(self, config):
        # Load configuration settings
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_pipeline()

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser(description="PyTorch Training Pipeline")

        parser.add_argument("--dataset", type=str, default=os.getenv("DATASET", "CIFAR100"),
                            choices=["MNIST", "CIFAR10", "CIFAR100"], help="Dataset to use")
        parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "resnet18"),
                            choices=["resnet18", "preactresnet18", "mlp", "lenet"], help="Model name to use")

        parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 50)), help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 64)), help="Batch size")
        parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE", 0.001)),
                            help="Learning rate")

        parser.add_argument("--optimizer", type=str, default=os.getenv("OPTIMIZER", "Adam"),
                            choices=["SGD", "Adam", "AdamW", "RMSprop"], help="Optimizer type")
        parser.add_argument("--scheduler", type=str, default=os.getenv("SCHEDULER", "StepLR"),
                            choices=["StepLR", "ReduceLROnPlateau", "None"], help="Learning rate scheduler")
        
        parser.add_argument("--use_augmentation", action="store_true", default=os.getenv("USE_AUGMENTATION", "False") == "True",
                            help="Apply data augmentation")
        parser.add_argument("--early_stopping", action="store_true", default=os.getenv("EARLY_STOPPING", "False") == "True",
                            help="Enable early stopping")
        parser.add_argument("--use_wandb", action="store_true", default=os.getenv("USE_WANDB", "False") == "True",
                            help="Enable Weights & Biases logging")
        parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "training_pipeline_project"),
                            help="Weights & Biases project name")

        parser.add_argument("--model_checkpoint", action="store_true", default=os.getenv("MODEL_CHECKPOINT", "False") == "True",
                            help="Save model checkpoints")
        parser.add_argument("--save_path", type=str, default=os.getenv("SAVE_PATH", "./model_checkpoint.pth"),
                            help="Path to save model checkpoints")

        args = parser.parse_args()

        config = {
            "dataset": args.dataset,
            "data_path": "./data",
            "batch_size": args.batch_size,
            "model_name": args.model_name,
            "optimizer": {
                "type": args.optimizer,
                "params": {"lr": args.learning_rate}
            },
            "scheduler": {
                "type": args.scheduler,
                "params": {"step_size": 10, "gamma": 0.1} if args.scheduler == "StepLR" else {}
            } if args.scheduler != "None" else None,
            "epochs": args.epochs,
            "early_stopping": args.early_stopping,
            "use_augmentation": args.use_augmentation,
            "augmentations": ["flip", "crop"] if args.use_augmentation else [],
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "use_tqdm": True,
            "model_checkpoint": args.model_checkpoint,
            "save_path": args.save_path
        }
        return PipelineWrapper(config)

    def setup_data(self):
        if self.config['dataset'] == 'MNIST':
            transform = self.get_augmentation_transforms()
            self.train_dataset = torchvision.datasets.MNIST(
                root=self.config['data_path'], train=True, transform=transform, download=True
            )
            self.test_dataset = torchvision.datasets.MNIST(
                root=self.config['data_path'], train=False, transform=transforms.ToTensor(), download=True
            )
        elif self.config['dataset'] in ['CIFAR10', 'CIFAR100']:
            transform = self.get_augmentation_transforms()
            dataset_class = torchvision.datasets.CIFAR10 if self.config['dataset'] == 'CIFAR10' else torchvision.datasets.CIFAR100
            self.train_dataset = dataset_class(
                root=self.config['data_path'], train=True, transform=transform, download=True
            )
            self.test_dataset = dataset_class(
                root=self.config['data_path'], train=False, transform=transforms.ToTensor(), download=True
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True
        )

    def get_augmentation_transforms(self):
        aug_transforms = []
        if self.config['use_augmentation']:
            if 'flip' in self.config['augmentations']:
                aug_transforms.append(transforms.RandomHorizontalFlip())
            if 'crop' in self.config['augmentations']:
                aug_transforms.append(transforms.RandomCrop(32, padding=4))
        aug_transforms.append(transforms.ToTensor())
        return transforms.Compose(aug_transforms)

    def setup_model(self):
        if self.config['dataset'] in ['CIFAR10', 'CIFAR100']:
            if self.config['model_name'] == 'resnet18':
                self.model = create_model('resnet18', pretrained=False, num_classes=10)
            elif self.config['model_name'] == 'preactresnet18':
                self.model = PreActResNet18() 
        elif self.config['dataset'] == 'MNIST':
            if self.config['model_name'] == 'mlp':
                self.model = MLP() 
            elif self.config['model_name'] == 'lenet':
                self.model = LeNet() 
        self.model.to(self.device)

    def setup_optimizer(self):
        optimizer_class = getattr(torch.optim, self.config['optimizer']['type'])
        self.optimizer = optimizer_class(self.model.parameters(), 
        **self.config['optimizer']['params'])

    def setup_scheduler(self):
        if self.config['scheduler']:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config['scheduler']['type'])
            self.scheduler = scheduler_class(self.optimizer, **self.config['scheduler']['params'])
        else:
            self.scheduler = None

    def setup_pipeline(self):
        self.pipeline = TrainingPipeline({
            "model": self.model,
            "train_loader": self.train_loader,
            "test_loader": self.test_loader,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "criterion": torch.nn.CrossEntropyLoss(),
            "epochs": self.config['epochs'],
            "early_stopping": self.config['early_stopping'],
            "early_stopping_criterion": self.config.get('early_stopping_criterion'),
            "training_step": self.config.get('training_step'),
            "testing_step": self.config.get('testing_step'),
            "use_wandb": self.config['use_wandb'],
            "project_name": self.config['wandb_project'],
            "use_tqdm": self.config['use_tqdm'],
            "model_checkpoint": self.config['model_checkpoint'],
            "model_checkpoint_criterion": self.config.get('model_checkpoint_criterion'),
            "save_path": self.config['save_path']
        })

    def train(self):
        if self.config['use_wandb']:
            wandb.init(project=self.config['wandb_project'])
            wandb.config.update(self.config) 
        self.pipeline.train()

    def parameter_sweep(self, sweep_config):
        sweep_id = wandb.sweep(sweep_config, project=self.config['wandb_project'])
        wandb.agent(sweep_id, function=self.train, count=sweep_config['count'])



wrapper = PipelineWrapper.from_args()
wrapper.train()
