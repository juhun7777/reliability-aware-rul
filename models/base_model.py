"""
Base Model Abstract Class
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins
"""
from abc import ABC, abstractmethod
import torch
import numpy as np


class PhysicsInformedLoss(torch.nn.Module):
    """
    Physics-Informed Loss Function
    - MSE Loss: Basic prediction error
    - Boundary Constraint: RUL must be in [0, 1] range
    """
    def __init__(self, lambda_monotonicity=0.1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.lambda_monotonicity = lambda_monotonicity

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        boundary_loss = torch.mean(torch.relu(pred - 1.0) + torch.relu(-pred))
        total_loss = mse_loss + self.lambda_monotonicity * boundary_loss
        return total_loss


class BaseModel(ABC):
    """Base class for all RUL prediction models"""

    def __init__(self, config, model_type='deep_learning'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.device = config.get_device() if model_type == 'deep_learning' else None

    @abstractmethod
    def build_model(self):
        """Build model architecture"""
        pass

    @abstractmethod
    def fit(self, train_data, val_data=None):
        """Train model"""
        pass

    @abstractmethod
    def predict(self, data):
        """Make predictions"""
        pass

    @abstractmethod
    def save(self, path):
        """Save model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load model"""
        pass

    def get_model_name(self):
        """Return model name"""
        return self.__class__.__name__


class BaseDeepLearningModel(BaseModel):
    """Base class for deep learning models"""

    def __init__(self, config):
        super().__init__(config, model_type='deep_learning')
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def _train_one_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(pred.detach().cpu().numpy().flatten())
            all_targets.extend(y.detach().cpu().numpy().flatten())

        avg_loss = total_loss / len(train_loader)
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def _validate(self, val_loader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)

                total_loss += loss.item()
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader)
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                pred = self.model(x)
                all_preds.extend(pred.cpu().numpy().flatten())

        return np.array(all_preds)

    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Model loaded from {path}")


class BaseMachineLearningModel(BaseModel):
    """Base class for machine learning models"""

    def __init__(self, config):
        super().__init__(config, model_type='machine_learning')

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def save(self, path):
        """Save model"""
        import joblib
        joblib.dump(self.model, path)
        print(f"  Model saved to {path}")

    def load(self, path):
        """Load model"""
        import joblib
        self.model = joblib.load(path)
        print(f"  Model loaded from {path}")
