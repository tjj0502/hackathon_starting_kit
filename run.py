"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import yaml
import os
from os import path as pt
from src.evaluations.evaluations import fake_loader, full_evaluation
import torch
from src.utils import get_experiment_dir, set_seed
from torch.utils.data import DataLoader, TensorDataset


def main(config):
    """
    Main interface, provides model the model training with target datasets and final assessment of trained model
    Parameters
    ----------
    config: configuration file
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    set_seed(config.seed)

    print(config)
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)

    training_set = TensorDataset(
            torch.load(config.data_dir + 'ref_data.pt').to(config.device).to(torch.float))

    train_dl = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    from src.baselines.models import get_trainer
    trainer = get_trainer(config, train_dl)

    # Create model directory and instantiate config.path
    get_experiment_dir(config)

    """
    Model training
    """
    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)
        trainer.save_model_dict()

    elif config.pretrained:
        pass

    """
    Model Evaluation
    """
    # Create the generative model, load the parameters and do evaluation
    from src.baselines.models import GENERATORS

    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim, n_layers=config.G_num_layers, init_fixed=config.init_fixed)
    generator.load_state_dict(torch.load(pt.join(
        config.exp_dir, 'generator_state_dict.pt')))

    # Use ref data to produce res_data
    test_set = TensorDataset(
        torch.load(config.data_dir + 'res_data.pt').to(config.device).to(torch.float))

    test_dl = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    fake_test_dl = fake_loader(generator, num_samples=len(test_dl.dataset),
                               n_lags=config.n_lags, batch_size=test_dl.batch_size, algo=config.algo
                               )

    res_dict = full_evaluation(test_dl, fake_test_dl, config)
    for k, v in res_dict.items():
        print(k, v)

if __name__ == '__main__':
    config_dir = 'configs/config.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    main(config)
