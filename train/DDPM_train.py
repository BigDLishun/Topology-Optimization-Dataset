from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import StepLR
import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from model import UNet,CNN
from  DDPM1 import DDPM
device = torch.device('cuda')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Define some hyperparameters
batch_size = 128
learning_rate = 0.01
num_epochs = 20

#path
path_random_input='data/random_input_data.npz'
path_random_target='data/random_target_data.npz'
path_cantilever_input='data/cantilever_beam_input_data.npz'
path_cantilever_target='data/cantilever_beam_target_data.npz'
path_continuous_input='data/continuous_beam_input_data.npz'
path_continuous_target='data/continuous_beam_target_data.npz'
path_simply_input='data/simply_supported_beam_input_data.npz'
path_simply_target='data/simply_supported_beam_target_data.npz'
random_best_model='./model/random_best_1000.ckpt'
random_final_model='./model/random_model_1000.ckpt'
cantilever_best_model='./model/cantilever_best.ckpt'
cantilever_final_model='./model/cantilever_model.ckpt'
continuous_best_model='./model/continuous_best.ckpt'
continuous_final_model='./model/continuous_model.ckpt'
simply_best_model='./model/simply_best.ckpt'
simply_final_model='./model/simply_model.ckpt'

class CustomLoss(nn.Module):
    def __init__(self, alpha):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # 计算 MAE 和 MSE
        loss_mae = F.l1_loss(y_pred, y_true)
        loss_mse = F.mse_loss(y_pred, y_true)

        # 计算加权平均损失
        loss = self.alpha * loss_mae + (1 - self.alpha) * loss_mse
        return loss

class CustomDataset(Dataset):
    def __init__(self, input_data, target_data, transform=None):
        self.input_data = input_data.astype(np.float32)
        self.target_data = target_data.astype(np.float32)
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]
        x = to_pil_image(x)
        y = to_pil_image(y)
        if self.transform:
            x = self.transform(x)
            x = self.transform1(x)
            y = self.transform(y)
        return x, y

# Define data transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomRotation([180, 180]),
    transforms.ToTensor(),
])
transform1 = transforms.Compose([
    transforms.Normalize((0.2,), (0.2,)),
])

# Load the input and target data
with np.load('data/random_input_data.npz') as data:
    input_data = data['input']
with np.load('data/random_target_data.npz') as data:
    target_data = data['target']

dataset = CustomDataset(input_data, target_data,transform=transform)
dataset_size = len(dataset)
print("dataset_size:",dataset_size)

train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1

train_size = int(train_ratio * dataset_size)
valid_size = int(valid_ratio * dataset_size)
test_size  = int(test_ratio * dataset_size)

torch.manual_seed(0)  # 设置随机种子，以便获得可重现的结果
train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
print('train_set:', len(train_set), '   val_set:', len(valid_set),'   test_set:', len(test_set))

# Create data loaders for training and validation and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


# Create a model and optimizer
#model = CNN().double()
num_timesteps = 100
segments = [(0, 0.02), (30, 0.01), (70, 0.005), (100, 0)]
model = DDPM(num_timesteps, segments).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 定义损失函数
#criterion = torch.nn.MSELoss().to(device)
criterion = CustomLoss(alpha=0.2).to(device)


# 训练模型
# Train the model
def training():
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train_loss = []
    lr_list, epoch_list = list(), list()
    total_step = 0
    best_acc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        t0 = time.time()
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            # 将数据放到 GPU 上（如果可用）
            inputs = x.to(device)
            labels = y.to(device)
            # 前向传播
            t = torch.randint(0, num_timesteps, (x.size(0),)).to(device)  # Randomly sample a timestep
            outputs = model(inputs,t)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step += 1
            train_loss.append(loss.item())

        t1 = time.time()
        scheduler.step()

            # 打印统计信息
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        print('Train Epoch time: ', (t1 - t0))
        # 保存模型的检查点
        torch.save(model.state_dict(), './model/DDPM.ckpt')
        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc >= best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), './model/DDPM_best.ckpt')
        print('best acc:', best_acc, 'best epoch:', best_epoch)
        print()


    plt.plot(train_loss,label="train_loss")
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('random_train_loss.png')
    plt.close()

    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.savefig("random_lr_curve.png")
    print('Finished Training')


def iou(pred, target):
    count=0
    pred = (pred > 0.5).int()    # 将浮点数转换为0或1的整数
    target = target.int()        # 确保目标值是整数
    intersection = (pred & target).float().sum((2, 3))
    union = (pred | target).float().sum((2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    count += (iou > 0.90).sum().item()

    return iou.mean(),count


#准确率
def evalute(model, loader):
    print("val start")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalacc=0
        totalacc1=0
        for x, y in loader:
            inputs = x.to(device)
            labels = y.to(device)
            t = torch.zeros(x.size(0), dtype=torch.long).to(device)
            output = model(inputs, t).to(device)
            acc,acc1 = iou(output, y.to(device))#.detach().cpu()
            totalacc+=acc
            totalacc1 += acc1
            #二值化
            th=0.5
            predicted = output.data
            pred_data=predicted
            pred_data[predicted>th]=1.0
            pred_data[predicted<=th]=0.001
            total += labels.size(0)
            correct += (pred_data == labels).sum().item()
        print("Val IOU:",totalacc/len(test_loader))
        print("IOU>0.9:",totalacc1/5000)
        return (100 * (correct / (total * 32* 64)))

# load model


def test():
    weight = torch.load('./model/DDPM.ckpt')
    model.load_state_dict(weight)
    print("test start")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalacc=0
        totalacc1=0
        for x, y in test_loader:
            inputs = x.to(device)
            labels = y.to(device)
            t = 0 # Test with no noise (t=0)
            output = model(inputs, t).to(device)
            acc,acc1 = iou(output, labels)#.detach().cpu()

            totalacc+=acc
            totalacc1 += acc1

            #二值化
            th=0.5
            predicted = output.data
            pred_data=predicted
            pred_data[predicted>th]=1.0
            pred_data[predicted<=th]=0.001
            total += labels.size(0)
            # output = output.squeeze(1).permute(1, 2, 0).numpy()
            # y=y.squeeze(0).permute(1, 2, 0).numpy()
            # output = (output - output.min()) / (output.max() - output.min())
            # plt.imshow(y, cmap='Greys')
            # plt.show()
            # # 显示图片
            # plt.imshow(output,cmap='Greys')
            # plt.show()
            correct += (pred_data == labels).sum().item()
            #break;

        print("Test IOU:",totalacc/len(test_loader))
        print("IOU>0.9",totalacc1/5000)
        print('Acc: {} %'.format(100 *(correct / (total*32*64))))


if __name__ == '__main__':
    #training()
    test()
    print("over")


