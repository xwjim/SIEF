import time
import matplotlib
import torch
from torch import nn
from torch import optim
import numpy as np
import random
from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from transformers.optimization import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from test import test
from utils import Accuracy, get_cuda, logging, print_params, Metrics
from models.SIEF import SentenceFocus
from models.HeterGSAN import HeterGSAN_GloVe, HeterGSAN_BERT

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']
def train(opt):
    rel2id = json.load(open(os.path.join(opt.data_dir, 'rel2id.json'), "r"))
    id2rel = {v: k for k, v in rel2id.items()}
    word2id = json.load(open(os.path.join(opt.data_dir, 'word2id.json'), "r"))
    ner2id = json.load(open(os.path.join(opt.data_dir, 'ner2id.json'), "r"))
    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                                   instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, opt,batch_size=opt.batch_size, shuffle=True,max_length=512)
        dev_loader = DGLREDataloader(dev_set, opt,batch_size=opt.test_batch_size, dataset_type='dev',max_length=512)

        if "HeterGSAN" in opt.model_name:
            model = HeterGSAN_BERT(opt)
        else:
            raise("Error")

    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        dev_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                               instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set,opt, batch_size=opt.batch_size, shuffle=True,max_length=512)
        dev_loader = DGLREDataloader(dev_set,opt, batch_size=opt.test_batch_size, dataset_type='dev',max_length=512)

        if "HeterGSAN" in opt.model_name:
            model = HeterGSAN_GloVe(opt)
        else:
            raise("Error")
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    print(model.parameters)
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '' and opt.load_model:
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))

    model = get_cuda(model)
    if opt.use_wandb:
        import wandb
        wandb.watch(model)

    if opt.use_model == 'bert':
        bert_param_ids = list(map(id, model.bert.parameters()))
        base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': opt.bert_lr},
            {'params': base_params}
            ], lr=lr, eps=1e-6)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                weight_decay=opt.weight_decay)

    Rel_bce = nn.BCEWithLogitsLoss(reduction='none')

    num_step_per_epoch = 3053//opt.batch_size
    total_steps = num_step_per_epoch*opt.epoch
    if opt.coslr:
        if opt.use_model == 'bert' or opt.use_model == 'roberta':
            scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=opt.warmup_epoch*num_step_per_epoch,num_training_steps=total_steps,num_cycles=1)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=opt.warmup_epoch*num_step_per_epoch,num_training_steps=total_steps,num_cycles=0.5)
    elif opt.steplr:
        if opt.use_model == 'bert':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20*num_step_per_epoch,40*num_step_per_epoch,60*num_step_per_epoch,80*num_step_per_epoch],gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30*num_step_per_epoch,60*num_step_per_epoch,90*num_step_per_epoch],gamma=0.5)
    elif opt.dynlr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",cooldown=10,threshold=0.005)
    elif opt.linearlr:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_epoch*num_step_per_epoch, num_training_steps=total_steps)

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    best_f1 = 0.0
    best_ign_f1 = 0.0
    best_epoch = 0
    best_theta = 0.0

    model.train()

    global_step = 0

    train_metric = Metrics("Re Train",logging,use_wandb=opt.use_wandb)
    logging('begin..')

    if opt.use_sief:
        Sief = SentenceFocus(opt)

    for epoch in range(start_epoch, opt.epoch + 1):
        start_time = time.time()

        train_metric.reset()

        for ii, d in enumerate(train_loader):
            relation_multi_label = d['relation_multi_label']
            relation_mask = d['relation_mask']

            output = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                context_ems_info=d['context_ems_info'],
                                h_t_pairs=d['h_t_pairs'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                sentence_id=d["context_sent"],
                                mention_id=d["context_mention"],
                                relation_mask=d['relation_mask'],
                                ht_pair_distance=d['ht_pair_distance'],
                                ht_sent_distance=d["ht_sent_distance"],
                                graph_adj=d['graph_adj'],
                                graph_info=d["graph_info"],
                                graph_node_num=d["graph_node_num"],
                                relation_path=d["relation_path"],
                                )

            predictions = output["predictions"]
            if opt.no_na_loss:
                loss = torch.sum(Rel_bce(predictions[...,1:], relation_multi_label[...,1:]) * relation_mask.unsqueeze(2)) / ((opt.relation_nums-1) * torch.sum(relation_mask))
            else:
                loss = torch.sum(Rel_bce(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (opt.relation_nums * torch.sum(relation_mask))

            if opt.use_sief:
                mask_d = Sief.prepro_data(d)
                output_hat = model(words=mask_d['context_idxs'],
                                src_lengths=mask_d['context_word_length'],
                                mask=mask_d['context_word_mask'],
                                context_ems_info=mask_d['context_ems_info'],
                                h_t_pairs=mask_d['h_t_pairs'],
                                entity_type=mask_d['context_ner'],
                                entity_id=mask_d['context_pos'],
                                sentence_id=mask_d["context_sent"],
                                mention_id=mask_d["context_mention"],
                                relation_mask=mask_d['relation_mask'],
                                ht_pair_distance=mask_d['ht_pair_distance'],
                                ht_sent_distance=mask_d["ht_sent_distance"],
                                graph_adj=mask_d['graph_adj'],
                                graph_info=mask_d["graph_info"],
                                graph_node_num=mask_d["graph_node_num"],
                                relation_path=mask_d["relation_path"],
                                )

                predictions_hat = output_hat["predictions"]

                loss_sf = Sief.sentence_focusing(predictions,predictions_hat,mask_d['relation_mask'],relation_multi_label)
                loss = (loss + loss_sf)/2
            
            optimizer.zero_grad()
            loss.backward()

            ## Relation
            pre_re_label = (predictions>0).long()
            train_metric.record(loss,pre_re_label[...,1:],relation_multi_label[...,1:],relation_mask.unsqueeze(dim=-1))

            if opt.clip > 0:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
        
            if opt.coslr or opt.steplr or opt.linearlr:
                scheduler.step()
            elif opt.dynlr:
                scheduler.step(test_loss)

            global_step += 1

        train_loss,train_acc,train_recall,train_ign_f1 = train_metric.cal_metric(global_step,get_lr(optimizer),log=True)
        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - start_time))
        logging('-' * 89)

        if epoch % opt.test_epoch == 0:
            eval_start_time = time.time()
            model.eval()
            test_loss,ign_f1, f1,theta, pr_x, pr_y = test(model, dev_loader, model_name, id2rel=id2rel,lr_rate=get_lr(optimizer),global_step=global_step,config=opt)
            model.train()

            if f1 > best_f1:
                best_f1 = f1
            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                best_epoch = epoch
                best_theta = theta
                if opt.wandb_name == "":
                    model_prefix = "base"
                else:
                    model_prefix = opt.wandb_name
                path = os.path.join(checkpoint_dir, model_name + "_"+ model_prefix + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr': lr,
                    'best_ign_f1': best_ign_f1,
                    "best_f1": f1,
                    'best_theta': best_theta,
                    'best_epoch': epoch
                }, path)

            logging('| epoch {:3d} | time: {:5.2f}s | best epoch {:3d} Ign F1 {:5.3f}% F1 {:5.3f}% Theta {:8.5f}'.format(epoch, time.time() - eval_start_time,best_epoch,100*best_ign_f1,100*best_f1,best_theta))
            logging('-' * 89)

        if epoch % opt.save_model_freq == 0:
            path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'checkpoint': model.state_dict()
            }, path)

        if opt.coslr or opt.steplr or opt.linearlr:
           scheduler.step()
        elif opt.dynlr:
           scheduler.step(test_loss)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f F1 = %f" % (best_epoch, best_ign_f1,best_f1 ))
    print("Storing best result...")
    print("Finish storing")

if __name__ == '__main__':
    opt = get_opt()
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    print(json.dumps(opt.__dict__, indent=4))

    opt.data_word_vec = np.load(os.path.join(opt.data_dir, 'vec.npy'))
    train(opt)
