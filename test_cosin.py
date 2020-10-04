import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from scheduler import GradualWarmupScheduler

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
steps = 10
num_epoch = 500
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler)
lr_values=[]
for epoch in range(num_epoch):
    scheduler_warmup.step(epoch)
    # for idx in range(steps):
    #     scheduler.step()
    print(epoch, optimizer.param_groups[0]['lr'])
    
    # print('Reset scheduler')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    lr_values.append(optimizer.param_groups[0]['lr'])

    plt.figure(2)
    plt.plot(lr_values, color = 'b') #

    plt.legend(['learning rate scheduler'], loc='upper left')
        # plt.show()
    plt.savefig('plot_lr.png')