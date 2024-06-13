"""Network modules for pytorch models.

Functions
---------


Classes
---------
TorchModel(base.base_model.BaseModel)

"""

import torch
import numpy as np
from base.base_model import BaseModel
import torch.nn.functional as F


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )

def dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )

def dense_block(out_features, act_fun, in_features=None):
    if in_features is None:
        block = [
            dense_lazy_couplet(out_channels, act_fun)
            for out_channels, act_fun in zip([*out_features], [*act_fun])
        ]
        return torch.nn.Sequential(*block)
    else:
        block = [
            dense_couplet(in_features, out_features, act_fun)
            for in_features, out_features, act_fun in zip(
                [*in_features], [*out_features], [*act_fun]
            )
        ]
        return torch.nn.Sequential(*block)

class RescaleLayer:
    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x




class TorchModel(BaseModel):
    def __init__(self, config, target_mean=None, target_std=None):
        super().__init__()

        self.config = config

        assert (
            len(self.config["basic_act"])
            == len(self.config["filters"])
        )
        assert len(self.config["hiddens_block"]) == len(
            self.config["hiddens_block_act"]
        )

        if target_mean is None:
            self.target_mean = torch.tensor(0.0)
        else:
            self.target_mean = torch.tensor(target_mean)

        if target_std is None:
            self.target_std = torch.tensor(1.0)
        else:
            self.target_std = torch.tensor(target_std)

        # # # Longitude padding
        # self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])

        # Simple network block?
        self.L1 = torch.nn.Linear(in_features=config["hiddens_block_in"][0], 
                                  out_features=config["hiddens_block"][0],
                                  bias=True)
        self.L2 = torch.nn.Linear(in_features=config["hiddens_block_in"][1], 
                                  out_features=config["hiddens_block"][1],
                                  bias=True)
        self.final = torch.nn.Linear(in_features=config["hiddens_block"][1], 
                                  out_features=4,
                                  bias=True)

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Dense blocks
        self.denseblock_mu = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_sigma = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_gamma = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )
        self.denseblock_tau = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )

        # Final dense layer
        self.finaldense_mu = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_sigma = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_gamma = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )
        self.finaldense_tau = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

        # Rescaling layers
        self.rescale_mu = RescaleLayer(self.target_std, self.target_mean)
        self.rescale_sigma = RescaleLayer(torch.tensor(1.0), torch.log(self.target_std))
        self.rescale_tau = RescaleLayer(torch.tensor(0.0), torch.tensor(1.0))

        # Output layers
        self.output_mu = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_sigma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_gamma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_tau = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )

    def forward(self, input):
        # x = self.pad_lons(x)

        # concatenate three inputs RMM1, RMM2, N34 as x input
        # x = torch.stack((x1, x2, x3), dim = 1)

        # basic hidden layers
        x = self.L1(input)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.final(x)

        # x = self.flat(x)

        # # build mu_layers
        # x_mu = self.denseblock_mu(x)
        # x_mu = self.finaldense_mu(x_mu)
        # mu_out = self.output_mu(x_mu)

        # # build sigma_layers
        # x_sigma = self.denseblock_sigma(x)
        # x_sigma = self.finaldense_sigma(x_sigma)
        # sigma_out = self.output_sigma(x_sigma)

        # # build gamma_layers
        # x_gamma = self.denseblock_gamma(x)
        # x_gamma = self.finaldense_gamma(x_gamma)
        # gamma_out = self.output_gamma(x_gamma)

        # # build tau_layers
        # x_tau = self.denseblock_tau(x)
        # x_tau = self.finaldense_tau(x_tau)
        # tau_out = self.output_tau(x_tau)

        # rescaling layers
        mu_out = self.rescale_mu(x[:,0])
        sigma_out = self.rescale_sigma(x[:,1])
        sigma_out = torch.exp(x[:,2])
        tau_out = self.rescale_tau(x[:,3])

        # final output, concatenate parameters together
        x = torch.cat((mu_out, sigma_out, gamma_out, tau_out), dim=-1)

        return x

    def predict(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

        if (dataset is None) & (dataloader is None):
            raise ValueError("both dataset and dataloader cannot be done.")

        if (dataset is not None) & (dataloader is not None):
            raise ValueError("dataset and dataloader cannot both be defined. choose one.")

        if dataset is not None:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.to(device)
        self.eval()
        
        with torch.inference_mode():

            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                x1, x2, x3, target = (
                    data[0].to(device), # RMM1
                    data[1].to(device), # RMM2
                    data[2].to(device), # Nino34
                    target.to(device),  # Lagged Seattle Precip Anom
                )

                out = self(x1, x2, x3).to("cpu").numpy()
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
    