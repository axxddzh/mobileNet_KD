import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import MobileNetV2
from tqdm import tqdm
from mobile_backbone import MobileBackbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def loss_fn_kd(outputs, labels, teacher_outputs, alpha = 0.5,temperature = 4):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "/media/axxddzh/database"))  # get data root path
image_path = data_root + "/imagenet/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path+"train/ILSVRC2012_img_train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8,pin_memory = True)

validate_dataset = datasets.ImageFolder(root=image_path + "val/ILSVRC2012_img_val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8,pin_memory = True)

net_t = MobileNetV2(num_classes=1000)
net_s = MobileBackbone()
# load pretrain weights
model_weight_path = "./mobilenet_v2.pth"
pre_weights = torch.load(model_weight_path)

missing_keys, unexpected_keys = net_t.load_state_dict(pre_weights, strict=False)

# freeze features weights
for param in net_t.features.parameters():
    param.requires_grad = False

net_t.to(device)
net_s.to(device)


# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net_t.parameters(), lr=0.0001)

optimizer_s = optim.Adam(net_s.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './MobileNetV2_S.pth'

# validate
net_t.eval()
acc = 0.0  # accumulate accurate number / epoch
with torch.no_grad():
    for val_data in tqdm(validate_loader):
        val_images, val_labels = val_data
        outputs = net_t(val_images.to(device))  # eval model only have last output layer
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += (predict_y == val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print('test_accuracy: %.3f' %
          (val_accurate))


for epoch in range(10):
    # train
    net_t.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # optimizer.zero_grad()
        # logits = net_t(images.to(device))
        # loss = loss_function(logits, labels.to(device))
        # loss.backward()
        # optimizer.step()
        #
        # # print statistics
        # running_loss += loss.item()
        # # print train process
        # rate = (step+1)/len(train_loader)
        # a = "*" * int(rate * 50)
        # b = "." * int((1 - rate) * 50)
        # print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")

        optimizer_s.zero_grad()
        out_t = net_t(images.to(device))
        logits_s = net_s(images.to(device))
        loss_s = loss_fn_kd(logits_s, labels.to(device),out_t)
        loss_s.backward()
        optimizer_s.step()

        # print statistics
        running_loss += loss_s.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss_s), end="")

    print()

    # validate
    net_s.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net_s(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net_s.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))



print('Finished Training')

