"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import yaml
import os
from os import path as pt
from src.evaluations.evaluations import fake_loader, full_evaluation
import torch
import pickle
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

    with open(config.data_dir + "ref_data.pkl", "rb") as f:
        loaded_array = pickle.load(f)
    training_set = torch.tensor(loaded_array).to(config.device).to(torch.float)

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
        input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim,
        n_layers=config.G_num_layers, init_fixed=config.init_fixed)
    generator.load_state_dict(torch.load(pt.join(
        config.exp_dir, 'generator_state_dict.pt')))

    # Load validation dataset
    with open(config.data_dir + "val_data.pkl", "rb") as f:
        validation_set = pickle.load(f)

    generator.eval()
    with torch.no_grad():
        fake_dataset = generator(len(validation_set), config.n_lags,
                                 device='cpu').numpy()

    # Save fake dataset
    with open(config.data_dir + "fake_data.pkl", "wb") as f:
        pickle.dump(fake_dataset, f)

    res_dict = full_evaluation(validation_set, fake_dataset, config)
    for k, v in res_dict.items():
        print(k, v)

if __name__ == '__main__':
    config_dir = 'configs/config.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    main(config)
