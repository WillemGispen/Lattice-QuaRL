# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%

def train(system, loss_fn):

    import sys
    sys.path.append('../')
    import torch
    # torch.autograd.set_detect_anomaly(True)


    # %%
    import lattice
    from trainer import Trainer

    # %% [markdown]
    # ### Make hparams from default_hparams

    # %%
    hparams = lattice.Q_DEFAULT_HPARAMS


    # %%
    if system == 'ising':
        hparams.potential = 'ising_potential'
        hparams.passive_rates = 'ising_passive_rates'
        # hparams.J = 0.32424925229
        # hparams.lattice_size = 4
        hparams.J = 0.32758326752  # Hamer pseudo-critical point for 6x6 # E0 goal: -1.063752757694
        hparams.lattice_size = 6
        hparams.g = 1.0
        hparams.E = -1.0
        hparams.validate_fn = 'local_stoquastic_energy'

    if system == 'xy':
        hparams.J = 1.0
        hparams.lattice_size = 4
        hparams.potential = 'xy_potential'
        hparams.passive_rates = 'xy_passive_rates'
        hparams.E = -1.0
        hparams.validate_fn = 'local_stoquastic_energy'
        # hparams.num_inner_layers = 5

    if system == 'heis':
        hparams.potential = 'heisenberg_antiferromagnet_potential'
        hparams.passive_rates = 'heisenberg_antiferromagnet_passive_rates'
        hparams.J = 1.0
        hparams.lattice_size = 4
        hparams.kernel_size = 3
        hparams.E = 0.0
        hparams.validate_fn = 'local_stoquastic_energy'
        hparams.num_inner_layers = 5


    # %%
    hparams.loss_fn = 'bellman_' + loss_fn

    if hparams.loss_fn == 'bellman_terminal':
        hparams.terminal = system + '_terminal'

    hparams.concentration = 0.5001

    hparams.lr = 3e-4
    hparams.passive_mcmc = False
    hparams.sync_rate = 20
    hparams.batch_size = 1024
    hparams.buffer_capacity = 64 * hparams.batch_size

    if hparams.loss_fn == 'bellman_discrete':
        hparams.tau = hparams.lattice_size**2
    elif hparams.loss_fn == 'bellman_continuous':
        hparams.tau = 1e-4
 
    hparams.gamma_lr = 1.0
    hparams.hidden_channels = 32
    hparams.num_steps = 64
    hparams.epoch_size = 10


    # %%
    print(hparams)


    # %%
    model = lattice.QModel(hparams)

    # # #%% Load model
    # checkpoint_path = 'results/heis/version_17/_ckpt_epoch_27.ckpt'
    # checkpoint_path = 'results/xy/version_28/_ckpt_epoch_99.ckpt'
    # checkpoint_path = 'results/ising/version_5/_ckpt_epoch_20.ckpt'
    # checkpoint_path = 'results/ising/version_' + f'{n}' + '/_ckpt_epoch_49.ckpt'
    # model2 = lattice.QModel.load_from_checkpoint(checkpoint_path)
    # model.load_state_dict(model2.state_dict())
    # model.eval()
    # trainer = Trainer(name=system, gpus=[3], max_epochs=100, nb_sanity_val_steps=1,
    #                   version=29, resume_from_checkpoint=checkpoint_path)


    # %% [markdown]
    # ## Train model

    # %%
    if torch.cuda.is_available():
        trainer = Trainer(name=system, gpus=[0], max_epochs=50, nb_sanity_val_steps=1)
    else:
        trainer = Trainer(name=system, max_epochs=50, nb_sanity_val_steps=1)

    # %%
    trainer.fit(model)



import argparse

parser = argparse.ArgumentParser(description='Do training')
parser.add_argument('system', metavar='system', type=str,
                    help='name of system')
parser.add_argument('loss', metavar='loss', type=str,
                    help='type of bellman loss')

args = parser.parse_args()
train(args.system, args.loss)
