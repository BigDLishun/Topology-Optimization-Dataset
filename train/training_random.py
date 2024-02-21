import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from torchvision import transforms, utils
import os
import time
import matplotlib.pyplot  as plt
from   utils import plot_image, plot_curve, one_hot
from torch.optim.lr_scheduler import StepLR
device = torch.device('cpu')
from model import UNet,CNN
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Define some hyperparameters
batch_size = 1
learning_rate = 0.01
num_epochs = 50



class CustomDataset(Dataset):
    def __init__(self, input_data, target_data, transform=None):
        self.input_data = input_data
        self.target_data = target_data
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform1(y)
        return x, y

# Define data transformations for data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2,), (0.2,))
])
transform1 = transforms.Compose([
    transforms.ToTensor(),
])

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

# Load the input and target data
with np.load('data/continuous_beam_input_data.npz') as data:
    input_data = data['input']
with np.load('data/continuous_beam_target_data.npz') as data:
    target_data = data['target']

# Split the data into training and validation sets
num_train = int(0.8 * len(input_data))
num_val = int(0.1 * len(input_data))
num_test =int(0.1 * len(input_data))

train_input = input_data[:40000]
train_target = target_data[:40000]
val_input = input_data[40000:45000]
val_target = target_data[40000:45000]
test_input = input_data[45000:50000]
test_target = target_data[45000:50000]
# train_input = input_data[:8000]
# train_target = target_data[:8000]
# val_input = input_data[8000:9000]
# val_target = target_data[8000:9000]
# test_input = input_data[9000:10000]
# test_target = target_data[9000:10000]


# Create data loaders for training and validation sets
train_dataset = CustomDataset(train_input, train_target, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(val_input, val_target, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CustomDataset(test_input, test_target, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print(len(train_input))
print(len(val_input))
print(len(test_input))
# Create a model and optimizer
#model = CNN().double()
model = UNet().double().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 定义损失函数
criterion = torch.nn.MSELoss().to(device)



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
            outputs = model(inputs)
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

        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc >= best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), './model/random_best_64_128_10000.ckpt')
        print('best acc:', best_acc, 'best epoch:', best_epoch)
        print()

        # 保存模型的检查点
        torch.save(model.state_dict(), './model/random_final_64_128_10000.ckpt')
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

def accuracy(pred, target, threshold=0.5):
    # pred = (pred > threshold).float()
    # target = target.float()
    iou_val = iou(pred, target)
    accuracy = (iou_val > 0.90).float().mean()
    return accuracy


#准确率
def evalute(model, loader):
    print("验证开始")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        totalacc=0
        totalacc1=0
        for x, y in loader:
            inputs = x.to(device)
            labels = y.to(device)
            output = model(inputs).to(device)
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
        return (100 * (correct / (total * 128 * 64)))

# load model


def test():
    weight = torch.load('./model/continuous_best.ckpt')
    model.load_state_dict(weight)
    print("导入测试模型成功")
    model.eval()
    print(model)
    with torch.no_grad():
        correct = 0
        total = 0
        totalacc=0
        totalacc1=0
        for x, y in test_loader:
            print("开始")
            inputs = x.to(device)
            labels = y.to(device)
            output = model(inputs).to(device)
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
            # 显示图片
            x1 = x.squeeze(0).permute(1, 2, 0).numpy()
            plt.imshow(x1, cmap='Greys')
            plt.show()
            y1=y.squeeze(0).permute(1, 2, 0).numpy()
            plt.imshow(y1, cmap='Greys')
            plt.show()
            output = output.squeeze(1).permute(1, 2, 0).numpy()
            output = (output - output.min()) / (output.max() - output.min())
            plt.imshow(output,cmap='Greys')
            plt.show()
            correct += (pred_data == labels).sum().item()
            break;
        print('Accuracy of the model on the test set: {} %'.format(100 *(correct / (total*32*64))))
        print("testdataset IOU:",acc)
        print("IOU>0.9",totalacc1/5000)


if __name__ == '__main__':
    #training()
    test()
    print("over")


