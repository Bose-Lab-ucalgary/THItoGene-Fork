# coding:utf-8 
import random

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEST1K
from predict import model_predict
from utils import *
from vis_model import THItoGene


def train(test_sample_ID=0, vit_dataset=ViT_HEST1K, epochs=300, modelsave_address="model", dataset_name="hest1k", gene_list = "3CA"):
    if dataset_name == "hest1k":
        if gene_list == "3CA":
            n_genes = 2977
        elif gene_list == "Hallmark":
            n_genes = 4376
        else:
            n_genes = 0
        tagname = gene_list + '_' + dataset_name + '_' + str(n_genes)
        model = THItoGene(n_genes = n_genes, learning_rate = 1e-5, route_dim = 64, caps = 20, heads = [16, 8], n_layers = 4)
        dataset = vit_dataset(mode='train', gene_list = gene_list)
        dataset_test = vit_dataset(mode = 'test', gene_list = gene_list)
    elif dataset_name == "her2st":
        tagname = "-htg_her2st_785_32_cv"
        model = THItoGene(n_genes=785, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=4)
        dataset = vit_dataset(train=True, fold=test_sample_ID)
        dataset_test = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    else:
        tagname = "-htg_skin_12_cv"
        model = THItoGene(n_genes=171, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=8)
        dataset = vit_dataset(train=True, fold=test_sample_ID)
        dataset_test = vit_dataset(train=False, sr=False, fold=test_sample_ID)

    mylogger = CSVLogger(save_dir=modelsave_address + "/../logs/",
                         name="my_test_log_" + tagname + '_' + str(test_sample_ID))
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=epochs,
                         logger=mylogger)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    trainer.fit(model, train_loader)

    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)

    pred, gt = trainer.predict(model=model, dataloaders=test_loader)[0]
    R, p_val = get_R(pred, gt)
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    print('Mean Pearson Correlation:', np.nanmean(R))
    trainer.save_checkpoint(modelsave_address+"/"+"last_train_"+tagname+'_'+str(test_sample_ID)+".ckpt")


def test(test_sample_ID=0, vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", gene_list = "3CA"):
    if dataset_name == "her2st":
        tagname = "-htg_her2st_785_32_cv"
        g = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        model = THItoGene.load_from_checkpoint(
            model_address + "/last_train_" + tagname + '_' + str(test_sample_ID) + ".ckpt", n_genes=785,
            learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8],
            n_layers=4)
        dataset = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    elif dataset_name == "hest1k":
        if gene_list == "3CA":
            n_genes = 2977
        elif gene_list == "Hallmark":
            n_genes = 4376
        else:
            n_genes = 0
        tagname = gene_list + '_' + dataset_name + '_' + str(n_genes)
        dataset = ViT_HEST1K(mode='val', gene_list = gene_list)
        g = dataset.get_gene_names()
        # model = THItoGene.load_from_checkpoint(model_address+"/THItoGene_" + tagname + "_" + str(test_sample_ID) + ".ckpt", n_genes = n_genes, learning_rate = 1e-5, route_dim = 64, caps =20, heads=[16, 8], n_layers = 4)
        model = THItoGene.load_from_checkpoint(model_address+"/THItoGene_her2st_" + str(test_sample_ID) + ".ckpt", n_genes = n_genes, learning_rate = 1e-5, route_dim = 64, caps =20, heads=[16, 8], n_layers = 4)
        
    else:
        tagname = "-htg_skin_12_cv"
        g = list(np.load('/home/user/jiayuran/code/cond/THItoGene/data/skin_hvg_cut_1000.npy', allow_pickle=True))
        model = THItoGene.load_from_checkpoint(
            model_address + "/THItoGene_" + tagname + '_' + str(test_sample_ID) + ".ckpt", n_genes=171,
            learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8],
            n_layers=8)
        dataset = vit_dataset(train=False, sr=False, fold=test_sample_ID)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device=device)

    adata_pred.var_names = g
    sc.pp.scale(adata_pred)

    if test_sample_ID in [5, 11, 17, 23, 26,
                          30] and dataset_name == 'her2st':
        label = dataset.label[dataset.names[0]]
        return adata_pred, adata_truth, label
    else:
        return adata_pred, adata_truth


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train()
    # for i in range(0, 32):
    #     train(i, ViT_HER2ST, 300, "model", "her2st")

    # # for i in range(12):
    # #     train(i, ViT_SKIN, 300, "/home/user/jiayuran/code/cond/THItoGene/model", "skin")

    # for i in range(0, 32):
    #     dataset = 'her2st'
    #     test_sample = i
    #     if dataset == "her2st":
    #         if test_sample in [5, 11, 17, 23, 26, 30]:
    #             pred, gt, label = test(test_sample, ViT_HER2ST, "model",
    #                                    dataset)
    #         else:
    #             pred, gt = test(test_sample, ViT_HER2ST, "model", dataset)
    #         R, p_val = get_R(pred, gt)
    #         pred.var["p_val"] = p_val
    #         pred.var["-log10p_val"] = -np.log10(p_val)
    #         print('Mean Pearson Correlation:', np.nanmean(R))
    #     else:
    #         pred, gt = test(test_sample, ViT_SKIN, "model", dataset)
    #         R, p_val = get_R(pred, gt)
    #         pred.var["p_val"] = p_val
    #         pred.var["-log10p_val"] = -np.log10(p_val)
    #         print('Mean Pearson Correlation:', np.nanmean(R))
