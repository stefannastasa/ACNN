""" Training pipeline class """ 
import torch
import wandb
from functools import partial

class TrainingPipeline:
    """
    Training pipeline class
    """
    def __init__(self, config):
        self.config = config
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module     = config["model"]

        self.train_dl  = config["train_loader"]
        self.test_dl   = config["test_loader"]
        self.optimizer = config["optimizer"]
        self.scheduler = config["scheduler"]
        self.criterion = config["criterion"]
        self.epochs    = config["epochs"]

        self.training_step = (partial(config["training_step"], pipeline=self)
                                or self._default_training_step)
        self.testing_step  = (partial(config["testing_step"], pipeline=self)
                                or self._default_testing_step)
        self.early_stopping = config["early_stopping"]

        if self.early_stopping:
            self.early_stopping_criterion = partial(config["early_stopping_criterion"], pipeline=self)

        self.use_wandb = config["use_wandb"]

        if self.use_wandb:
            wandb.init(project=config["project_name"])
            wandb.watch(self.model)

        self.use_tqdm = config["use_tqdm"]

        if self.use_tqdm:
            self.training_bar = config["tqdm_training"]
            self.testing_bar  = config["tqdm_testing"]

        self.model_checkpoint = config["model_checkpoint"]

        if self.model_checkpoint:
            self.model_checkpoint_criterion = partial(config["model_checkpoint_criterion"], pipeline=self)
            self.save_path = config["save_path"]

    def _default_training_step(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        correct = (output.argmax(dim=1) == target).sum().item()
        return loss.item(), correct

    def _default_testing_step(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)

        correct = (output.argmax(dim=1) == target).sum().item()
        return loss.item(), correct


    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct = 0, 0

        for data, target in self.train_dl:
            loss, batch_correct = self.training_step(data, target)
            total_loss += loss
            correct += batch_correct

            if self.use_tqdm:
                self.training_bar.update(len(data))

        accuracy = 100.0 * correct / len(self.train_dl.dataset)
        if self.use_wandb:
            wandb.log({"train_loss": total_loss / len(self.train_dl), "train_accuracy": accuracy})

    @torch.inference_mode
    def _test_one_epoch(self, epoch):
        self.model.eval()
        total_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for data, target in self.test_dl:
                loss, batch_correct = self.testing_step(data, target)

                total_loss += loss
                correct    += batch_correct

                if self.use_tqdm:
                    self.testing_bar.update(len(data))

        accuracy = 100. * correct /len( self.test_dl.dataset )
        if self.use_wandb:
            wandb.log({"val_loss": total_loss / len(self.test_dl), "val_accuracy": accuracy})

        return total_loss / len(self.test_dl)

    def save_model(self, epoch, metric_value):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            'loss': metric_value
        }
        torch.save(checkpoint, self.save_path)

        print(f"Checkpoint saved at epoch {epoch} with metric value: {metric_value}")

    def train(self):
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)
            val_loss = self._test_one_epoch(epoch)

            if self.scheduler:
                self.scheduler.step(val_loss)

            if self.early_stopping:
                if self.early_stopping_criterion():
                    print("Early stopping triggered")
                    break
            if self.model_checkpoint:
                if self.model_checkpoint_criterion():
                    self.save_model(epoch, val_loss)
