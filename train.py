from train_template import TemplateModel
from tensorboardX import SummaryWriter
from model import NewFaceModel
import uuid
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from dataset import HelenDataset
from preprocess import Resize, ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

uuid = str(uuid.uuid1())[0:8]
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=6, type=int, help="Choose which GPU")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=20, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--momentum", default=0.9, type=int, help="momentum ")
parser.add_argument("--weight_decay", default=0.005, type=int, help="weight_decay ")
args = parser.parse_args()
print(args)
img_root_dir = "/data1/yinzi/datas"
# part_root_dir = "/data1/yinzi/facial_parts"
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}
device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
twostage_Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                                    root_dir=img_root_dir,
                                    transform=transforms.Compose([
                                        Resize((512, 512)),
                                        ToTensor(),
                                        Normalize()
                                    ])
                                    )
                    for x in ['train', 'val']
                    }

twostage_dataloader = {x: DataLoader(twostage_Dataset[x], batch_size=args.batch_size,
                                     shuffle=True, num_workers=20)
                       for x in ['train', 'val']
                       }


class ModelTrain(TemplateModel):
    def __init__(self):
        super(ModelTrain, self).__init__()
        self.writer = SummaryWriter('test2')

        self.model = NewFaceModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()

        self.train_loader = twostage_dataloader['train']
        self.eval_loader = twostage_dataloader['val']

        self.device = device

        self.ckpt_dir = "checkpoint_%s" % uuid
        self.display_freq = 10
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.mode = 'train'

    def train_loss(self, batch):
        x, y = batch['image'].to(device), batch['labels'].to(device)
        pred = self.model(x)
        # pred Shape(N, 11, 512, 512)
        # y Shape(N, 512 ,512)
        loss = self.criterion(pred, y)

        return loss, None

    def eval_error(self):
        error = 0
        iter = 0
        for i, batch in enumerate(self.eval_loader):
            iter += 1
            x, y = batch['image'].to(device), batch['labels'].to(device)
            pred = self.model(x)
            error += self.metric(pred, y).item()

        error /= iter

        return error, None


def start_train():
    train = ModelTrain()
    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


if __name__ == '__main__':
    start_train()
