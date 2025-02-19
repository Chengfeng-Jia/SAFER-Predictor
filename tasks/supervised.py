import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses

def percentile(t, q):
    k = 1 + round( float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        args=None,
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self.args=args
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def shared_step1(self, x):
        # (batch_size, seq_len/pre_len, num_nodes)
        #x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        #y = y.reshape((-1, y.size(2)))
        return predictions

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)
    def attack(self,batch):
        x,y=batch
        if self.args.soft:
            self.args.num_iter=1
        y = y.reshape((-1, y.size(2)))
        delta = torch.zeros_like(x, requires_grad=True)
        if not self.args.random:
            for t in range(self.args.num_iter):
                    predictions=self.shared_step1(x+delta)
                    loss = F.mse_loss(predictions, y)
                    loss.backward()
                    grad=delta.grad.detach()
                    grad=x.norm()*grad/grad.norm()
                    delta.data = (delta + self.args.alpha*grad)
                    delta.grad.zero_()
        else:
            delta=torch.randn_like(x)
            delta=x.norm()*delta/delta.norm()*self.args.alpha
        if self.args.soft:            
            if self.args.sparse:
                mask=torch.rand_like(x)<self.args.noise_ratio
                x_noise=x.clone().detach()+delta*mask
            else:
                x_noise=x.clone().detach()+delta

        else:
            k_val=percentile(torch.abs(delta),self.args.noise_ratio)
            x_noise=x.clone().detach()
            x_noise[x_noise<k_val]=0
        
        return x_noise

    def noise_data(self,x):
        mask=torch.rand(x.shape).to(x.device)
        mask=mask<self.args.noise_ratio
        max_value=torch.max(x)-torch.min(x)
        noise1=torch.distributions.Uniform(low=-1, high=1).sample(x[mask].shape).to(x.device)
        x[mask]=x[mask]+noise1*max_value*self.args.noise_sever*0.1
        return x
    def training_step(self, batch, batch_idx):
        if self.args.attack:
            noise_x=self.attack(batch)
        
        predictions, y = self.shared_step(batch, batch_idx)
        if self.args.attack:
            batch[0]=noise_x
            predictions_adv,y=self.shared_step(batch,batch_idx)
            loss=self.loss(predictions, y)+self.loss(predictions_adv, predictions)*self.args.lamda       
        else:    
            loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        parser.add_argument("--noise", action='store_true')
        parser.add_argument("--attack", action='store_true')
        parser.add_argument("--noise_ratio", type=float, default=0.05)
        parser.add_argument("--alpha", type=float, default=0.01)
        parser.add_argument("--lamda", type=float, default=0.5)
        parser.add_argument("--num_iter", type=int, default=1)

        #parser.add_argument("--noise_ratio_node",type=float,default=0.2)
        parser.add_argument("--noise_test", action='store_true')
        parser.add_argument("--soft", action='store_true')
        parser.add_argument("--random", action='store_true')
        parser.add_argument("--sparse", action='store_true')
        parser.add_argument("--direction", action='store_true')
        parser.add_argument("--noise_ratio_test", type=float, default=0.2)
        parser.add_argument("--noise_ratio_node_test",type=float,default=0.2)
        parser.add_argument("--noise_sever", type=float, default=1.0)
        parser.add_argument("--noise_type", type=str, default='gaussian',choices=['gaussian','missing'])
        return parser
