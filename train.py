import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import argparse
from os.path import join as pjoin
import time

from dataset import Mnist
from utils import Logger
import models


def main(args, logger):
    # ================ seed and device ===================
    np.random.seed(42)
    torch.manual_seed(42)
    if args.cuda:
        torch.cuda.manual_seed_all(42)
        device = 'cuda'
    else:
        device = 'cpu'
    # ================= data ====================
    mnist = Mnist(args.data_dir, mode='train')
    train_loader = torch.utils.data.DataLoader(mnist.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(mnist.val_dataset, batch_size=args.batch_size, shuffle=False)
    # ================== Model ===================
    model = models.get_model(name=args.model, n_class=mnist.n_class) 
    model.to(device)
    # ================== Loss, optimizer and scheduler ===============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                        patience=10, verbose=False, threshold=0.0001)
    # ================== Training and validation ===============
    start_epoch = 0
    if args.resume:
        logger.write("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint['epoch']
    train_loss_his = []
    val_loss_his = []
    val_acc_his = []
    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        # ================== Training ====================
        st = time.time()
        train_loss = 0
        for idx, (img, lab) in enumerate(train_loader):
            img = img.to(device)
            lab = lab.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (idx+1)%args.print_freq == 0:
                logger.write(f'Epoch: [{epoch}][{idx+1}/{len(train_loader)}]\t'
                             f'Loss: {loss.item():.4f}\t'
                             f"lr: {optimizer.param_groups[0]['lr']:.6f}\t")
        train_loss /= len(train_loader)
        train_cost = round(time.time() - st)
        # ===================== Testing ===================
        model.eval()
        st = time.time()
        val_acc = 0
        val_loss = 0
        best_acc = -100
        with torch.no_grad():
            for idx, (img, lab) in enumerate(val_loader):
                img = img.to(device)
                lab = lab.to(device)
                out = model(img)
                pred = out.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(lab.view_as(pred)).sum().item()
                loss = criterion(out, lab)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_loss_his.append(val_loss)
        val_acc_his.append(val_acc)
        # =================== adjust lr, save model and print log ================
        scheduler.step(val_loss)
        logger.write(f'Epoch: {epoch:2d} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')
        logger.write(f'Train cost: {train_cost}s Val cost: {round(time.time()-st)}s')
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_iou": best_acc,
            }
        save_path = pjoin(args.save_dir, f"{args.model}_best_model.pkl")
        torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Classification')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--data-dir', type=str, default='/home/jjou/sunjiahui/MLproject/dataset/MNIST')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default='saved')
    parser.add_argument('--print-freq', type=int, default=50)
    args = parser.parse_args()
    logger = Logger(pjoin(args.save_dir, args.model+'_train.log'))
    logger.write(f'Config: {args}')

    main(args, logger)
