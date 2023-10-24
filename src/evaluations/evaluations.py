import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.utils import loader_to_tensor, to_numpy
from src.evaluations.test_metrics import CrossCorrelLoss, HistoLoss, CovLoss, ACFLoss
import numpy as np

def _train_classifier(model, train_loader, test_loader, config, epochs=100):
    """
    Train a NN-based classifier to obtain the discriminative score
    Parameters
    ----------
    model: torch.nn.module
    train_loader: torch.utils.data DataLoader: dataset for training
    test_loader: torch.utils.data DataLoader: dataset for testing
    config: configuration file
    epochs: number of epochs for training

    Returns
    -------
    test_acc: model's accuracy in test dataset
    test_loss: model's cross-entropy loss in test dataset
    """
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    dataloader = {'train': train_loader, 'validation': test_loader}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    criterion = torch.nn.CrossEntropyLoss()
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        # print("Epoch {}/{}".format(epoch + 1, epochs))
        # print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)
                # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            # print("{} Loss: {:.4f} Acc: {:.4f}".format(
            #     phase, epoch_loss, epoch_acc))

            if phase == "validation" and epoch_acc >= best_acc:
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

    # print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_acc, test_loss = _test_classifier(
        model, test_loader, config)
    return test_acc, test_loss


def _test_classifier(model, test_loader, config):
    """
    Computes the test metric for trained classifier
    Parameters
    ----------
    model: torch.nn.module, trained model
    test_loader:  torch.utils.data DataLoader: dataset for testing
    config: configuration file

    Returns
    -------
    test_acc: model's accuracy in test dataset
    test_loss: model's cross-entropy loss in test dataset
    """
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    running_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    test_loss = running_loss / total
    print("Accuracy of the network on the {} test samples: {}".format(total, (100 * test_acc)))
    return test_acc, test_loss


def _train_regressor(model, train_loader, test_loader, config, epochs=100):
    """
    Training a predictive model to obtain the predictive score
    Parameters
    ----------
    model: torch.nn.module
    train_loader: torch.utils.data DataLoader: dataset for training
    test_loader: torch.utils.data DataLoader: dataset for testing
    config: configuration file
    epochs: number of epochs for training

    Returns
    -------

    """
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999
    dataloader = {'train': train_loader, 'validation': test_loader}
    criterion = torch.nn.L1Loss()

    for epoch in range(epochs):
        # print("Epoch {}/{}".format(epoch + 1, epochs))
        # print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            total = 0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(True):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    # print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
            epoch_loss = running_loss / total
            # print("{} Loss: {:.4f}".format(phase, epoch_loss))

        if phase == "validation" and epoch_loss <= best_loss:

            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Clean CUDA Memory
            del inputs, outputs, labels
            torch.cuda.empty_cache()
    # print("Best Val MSE: {:.4f}".format(best_loss))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_loss = _test_regressor(
        model, test_loader, config)

    return test_loss


def _test_regressor(model, test_loader, config):
    """
    Computes the test metric for trained classifier
    Parameters
    ----------
    model: torch.nn.module, trained model
    test_loader:  torch.utils.data DataLoader: dataset for testing
    config: configuration file

    Returns
    -------
    test_loss: L1 loss between the real and predicted paths by the model in test dataset
    """
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    total = 0
    running_loss = 0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            total += labels.size(0)

    test_loss = running_loss / total
    return test_loss


def fake_loader(generator, num_samples, n_lags, batch_size, algo, **kwargs):
    """
    Helper function that transforms the generated data into dataloader, adapted from different generative models
    Parameters
    ----------
    generator: nn.module, trained generative model
    num_samples: int,  number of paths to be generated
    n_lags: int, the length of path to be generated
    batch_size: int, batch size for dataloader
    config: configuration file
    kwargs

    Returns
    Dataload of generated data
    -------

    """
    with torch.no_grad():
        if algo == 'TimeGAN':
            fake_data = generator(batch_size=num_samples,
                                  n_lags=n_lags, device='cpu')
            if 'recovery' in kwargs:
                recovery = kwargs['recovery']
                fake_data = recovery(fake_data)
        elif algo == 'TimeVAE':
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition).permute([0, 2, 1])
        else:
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition)
        tensor_x = torch.Tensor(fake_data)
    return DataLoader(TensorDataset(tensor_x), batch_size=batch_size)


def compute_discriminative_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config,
                                 hidden_size=64, num_layers=2, epochs=30, batch_size=512):

    def create_dl(real_dl, fake_dl, batch_size):
        train_x, train_y = [], []
        for data in real_dl:
            train_x.append(data[0])
            train_y.append(torch.ones(data[0].shape[0], ))
        for data in fake_dl:
            train_x.append(data[0])
            train_y.append(torch.zeros(data[0].shape[0], ))
        x, y = torch.cat(train_x), torch.cat(train_y).long()
        idx = torch.randperm(x.shape[0])

        return DataLoader(TensorDataset(x[idx].view(x.size()), y[idx].view(y.size())), batch_size=batch_size)

    train_dl = create_dl(real_train_dl, fake_train_dl, batch_size)
    test_dl = create_dl(real_test_dl, fake_test_dl, batch_size)

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                              hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_acc_list = []
    for i in range(1):
        model = Discriminator(
            train_dl.dataset[0][0].shape[-1], hidden_size, num_layers)

        test_acc, test_loss = _train_classifier(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs)
        test_acc_list.append(test_acc)
    mean_acc = np.mean(np.array(test_acc_list))
    std_acc = np.std(np.array(test_acc_list))
    return abs(mean_acc-0.5), std_acc


def compute_classfication_score(real_train_dl, fake_train_dl, config,
                                hidden_size=64, num_layers=3, epochs=100):
    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(Discriminator, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)
    model = Discriminator(
        real_train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=config.num_classes)
    TFTR_acc = _train_classifier(
        model.to(config.device), fake_train_dl, real_train_dl, config, epochs=epochs)
    TRTF_acc = _train_classifier(
        model.to(config.device), real_train_dl, fake_train_dl, config, epochs=epochs)
    return TFTR_acc, TRTF_acc


def compute_predictive_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config,
                             hidden_size=64, num_layers=3, epochs=100, batch_size=128):
    def create_dl(train_dl, test_dl, batch_size):
        x, y = [], []
        _, T, C = next(iter(train_dl))[0].shape

        T_cutoff = int(T/10)
        for data in train_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        for data in test_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        x, y = torch.cat(x), torch.cat(y),
        idx = torch.randperm(x.shape[0])
        dl = DataLoader(TensorDataset(x[idx].view(
            x.size()), y[idx].view(y.size())), batch_size=batch_size)

        return dl
    train_dl = create_dl(fake_train_dl, fake_test_dl, batch_size)
    test_dl = create_dl(real_train_dl, real_test_dl, batch_size)

    class predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_loss_list = []
    for i in range(1):
        model = predictor(
            train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=train_dl.dataset[0][1].shape[-1])
        test_loss = _train_regressor(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs)
        test_loss_list.append(test_loss)
    mean_loss = np.mean(np.array(test_loss_list))
    std_loss = np.std(np.array(test_loss_list))
    return mean_loss, std_loss

def full_evaluation(real_dl, fake_dl, config):
    """ evaluation for the synthetic generation, including.
        1) Stylized facts: marginal distribution, cross-correlation, autocorrelation, covariance scores.
        2) Implicit scores: discriminative score, predictive score.
    Args:
        real_dl (_type_): torch.dataloader
        fake_dl (_type_): torch..dataloader
    """
    d_scores = []
    p_scores = []
    hist_losses = []
    cross_corrs = []
    cov_losses = []
    acf_losses = []

    real_data = loader_to_tensor(real_dl).to(config.device).to(torch.float)
    fake_data = loader_to_tensor(fake_dl).to(config.device).to(torch.float)

    set_size = int(0.8 * real_data.shape[0])

    real_data_train = real_data[:set_size,:,:]
    real_data_test = real_data[set_size:,:,:]
    fake_data_train = fake_data[:set_size,:,:]
    fake_data_test = fake_data[set_size:,:,:]

    real_train_dl = DataLoader(TensorDataset(real_data_train), batch_size=128)
    real_test_dl = DataLoader(TensorDataset(real_data_test), batch_size=128)
    fake_train_dl = DataLoader(TensorDataset(fake_data_train), batch_size=128)
    fake_test_dl = DataLoader(TensorDataset(fake_data_test), batch_size=128)

    dim = real_data.shape[-1]

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    d_score_mean, d_score_std = compute_discriminative_score(
        real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, int(dim / 2), 1, epochs=10, batch_size=128)
    d_scores.append(d_score_mean)
    p_score_mean, p_score_std = compute_predictive_score(
        real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, 32, 2, epochs=10, batch_size=128)
    p_scores.append(p_score_mean)

    cross_corrs.append(to_numpy(CrossCorrelLoss(
        real_data, name='cross_correlation')(fake_data)))
    hist_losses.append(
        to_numpy(HistoLoss(real_data[:, 1:, :], n_bins=50, name='marginal_distribution')(fake_data[:, 1:, :])))
    acf_losses.append(
        to_numpy(ACFLoss(real_data, name='auto_correlation', stationary=False)(fake_data)))
    cov_losses.append(to_numpy(CovLoss(real_data, name='covariance')(fake_data)))

    d_mean, d_std = np.array(d_scores).mean(), np.array(d_scores).std()
    p_mean, p_std = np.array(p_scores).mean(), np.array(p_scores).std()

    hist_mean, hist_std = np.array(
        hist_losses).mean(), np.array(hist_losses).std()
    corr_mean, corr_std = np.array(
        cross_corrs).mean(), np.array(cross_corrs).std()
    cov_mean, cov_std = np.array(
        cov_losses).mean(), np.array(cov_losses).std()
    acf_mean, acf_std = np.array(
        acf_losses).mean(), np.array(acf_losses).std()

    result_dict = {
        "Predictive Score": p_mean,
        "Discriminative Score": d_mean,
        "Marginal Score": hist_mean,
        "Correlation Score": corr_mean,
        "Auto-correlation Score": acf_mean,
        "Covariance Score": cov_mean
    }

    return result_dict