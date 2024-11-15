
import torch
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from finetune import ViTLinear, inference, Trainer, VPT
import yaml
from torch.optim.lr_scheduler import MultiStepLR

from datasets import get_flower102


FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'vit_linear',
                    'The experiment with corresponding hyperparameters to run. See config.yaml')
flags.DEFINE_string('output_dir', 'run1', 'Output Directory')
flags.DEFINE_string('encoder', 'vit_b_32',
                    'Select the encoder from linear, resnet18, resnet50, vit_b_16, vit_b_32')
flags.DEFINE_string('data_dir', './flower-dataset-reduced', 'Directory with coco data')

def get_config(exp_name, encoder):
    dir_name = f'{FLAGS.output_dir}/runs-{encoder}-flower102/demo-{exp_name}'

    # add/modify hyperparameters of your class in config.yaml
    encoder_registry = {
        'ViTLinear': ViTLinear,
    }
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    wd = config['wd']
    epochs = config['epochs']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    momentum = config['momentum']
    net_class = encoder_registry[config['net_class']]
    batch_size = config['batch_size']
    milestones = config['milestones']
    gamma = config['gamma']

    return net_class, dir_name, (optimizer, lr, wd, momentum), (scheduler, epochs, milestones, gamma), batch_size


def main(_):
    torch.set_num_threads(2)
    torch.manual_seed(0)

    print(f"Running: {FLAGS.exp_name} with encoder {FLAGS.encoder} ------------------")

    net_class, dir_name, \
        (optimizer, lr, wd, momentum), \
        (scheduler, epochs, milestones, gamma), batch_size = \
        get_config(FLAGS.exp_name, FLAGS.encoder)

    train_data = get_flower102(FLAGS.data_dir,'train')
    val_data = get_flower102(FLAGS.data_dir,'val')
    test_data = get_flower102(FLAGS.data_dir,'test')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    tmp_file_name = dir_name + '/best_model.pth'
    # device = torch.device('cuda:0')
    # For Mac Users
    device = torch.device('mps')

    writer = SummaryWriter(f'{dir_name}/lr{lr:0.6f}_wd{wd:0.6f}', flush_secs=10)

    # we suggest creating a class called VITPrompt and putting your logic there.
    # Then just initialize your model from that class.
    # model = net_class(102, FLAGS.encoder)
    model = VPT(102, FLAGS.encoder)
    model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                      optimizer=optimizer, lr=lr, wd=wd, momentum=momentum,
                      scheduler=scheduler, epochs=epochs,
                      device=device)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"lr: {lr:0.7f}, wd: {wd:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")

    print("Training complete--------------------")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{dir_name}/best_model.pth'))
    inference(test_dataloader, model, device, result_path=dir_name + '/test_predictions.txt')


if __name__ == '__main__':
    app.run(main)
