from src.baselines.RCGAN import RCGANTrainer
from src.baselines.networks.discriminators import LSTMDiscriminator
from src.baselines.networks.generators import LSTMGenerator
import torch
from src.utils import loader_to_tensor
GENERATORS = {'LSTM': LSTMGenerator}

def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)


DISCRIMINATORS = {'LSTM': LSTMDiscriminator}


def get_discriminator(discriminator_type, input_dim, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, **kwargs)


def get_trainer(config, train_dl):
    # print(config)

    print(config.algo)

    x_real_train = loader_to_tensor(
        train_dl).to(config.device)

    config.input_dim = x_real_train.shape[-1]

    D_out_dim = 1
    return_seq = True

    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim,
        n_layers=config.G_num_layers, init_fixed=config.init_fixed)
    discriminator = DISCRIMINATORS[config.discriminator](
        input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=D_out_dim, n_layers=config.D_num_layers,
        return_seq=return_seq)
        # print('GENERATOR:', generator)
        # print('DISCRIMINATOR:', discriminator)

    trainer = RCGANTrainer(G=generator, D=discriminator,
                        train_dl=train_dl, batch_size=config.batch_size, n_gradient_steps=config.steps,
                        config=config)

    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    # Required for multi-GPU
    torch.backends.cudnn.benchmark = True

    return trainer
