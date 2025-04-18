import torch
from data_loader import MVTecADTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model import IGAFNet
from loss import SSIM
import os


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'IGAF'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = IGAFNet(in_channel=1, out_channel=3)
        model.cuda()
        model.apply(weights_init)
        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      ],weight_decay=0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_l1 = torch.nn.modules.loss.L1Loss()
        loss_ssim = SSIM()

        dataset = MVTecADTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        n_iter = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch + 1))
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                gray_grayimage=sample_batched["auggray"].cuda()
                gray_rec = model(gray_grayimage)
                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                loss = l2_loss + ssim_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                if args.visualize and n_iter % 400 == 0:
                    visualizer.visualize_image_batch(gray_grayimage, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')


                n_iter +=1

            scheduler.step()

            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))


if __name__=="__main__":
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
        parser.add_argument('--obj_id', action='store', type=int, required=True)
        parser.add_argument('--lr', action='store', type=float, required=True)
        parser.add_argument('--bs', action='store', type=int, required=True)
        parser.add_argument('--epochs', action='store', type=int, required=True)
        parser.add_argument('--data_path', action='store', type=str, required=True)
        parser.add_argument('--log_path', action='store', type=str, required=True)
        parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
        parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
        parser.add_argument('--visualize', action='store_true')

        args = parser.parse_args()

        obj_batch = [['carpet'],
                     ['grid'],
                     ['leather'],
                     ['tile'],
                     ['wood'],
                     ['bottle'],
                     ['cable'],
                     ['capsule'],
                     ['hazelnut'],
                     ['metal_nut'],
                     ['pill'],
                     ['screw'],
                     ['toothbrush'],
                     ['transistor'],
                     ['zipper']
                     ]

        if int(args.obj_id) == -1:
            obj_list = ['carpet',
                        'grid',
                        'leather',
                        'tile',
                        'wood',
                        'bottle',
                        'cable',
                        'capsule',
                        'hazelnut',
                        'metal_nut',
                        'pill',
                        'screw',
                        'toothbrush',
                        'transistor',
                        'zipper'
                        ]
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]

        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)


