from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model import Model
from dataset import KDDcupDataset


def main():
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / 'model').mkdir(exist_ok=True, parents=True)
    batch_size = 2048
    num_epochs = 10
    train_data_amount = 8 * 1e7 * 0.9
    test_data_amount = 8 * 1e7 * 0.1

    train_dataset = KDDcupDataset('../data/D80M_train.tsv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_dataset = KDDcupDataset('../data/D80M_test.tsv')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    keyword_processor_args = {'sequence_len': 16, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 32}
    title_processor_args = {'sequence_len': 32, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 64}
    description_processor_args = {'sequence_len': 50, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 100}
    query_processor_args = {'sequence_len': 128, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 256}
    soft_ordering_args = {'input_dim': 3*(32+64+100+256)+3*16+11, 'output_dim': 1, 'in_num_kernels': 256, 'chunk_size': 16}
    other_embedding_dim = 16
    model = Model(keyword_processor_args, title_processor_args, description_processor_args, 
                  query_processor_args, soft_ordering_args, other_embedding_dim)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        t_s = time()
        model.train()
        cum_loss = 0.0
        aucs = []
        for i, batch in enumerate(train_dataloader):
            target = batch['target'].cuda()
            input_keyword = batch['keyword'].cuda()
            input_title = batch['title'].cuda()
            input_description = batch['description'].cuda()
            input_query = batch['query'].cuda()
            input_advertiser = batch['advertiser'].cuda()
            input_ad = batch['ad'].cuda()
            input_display = batch['display'].cuda()
            input_num = batch['numeric'].cuda()
            output = model(input_keyword, input_title, input_description, input_query, input_advertiser, input_ad, input_display, input_num)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='sum')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = loss.detach().item()
            cum_loss += loss_value
            if i % 100 == 0 and i > 0:
                aucs.append(roc_auc_score(target.detach().cpu().numpy(), output.detach().cpu().numpy()))
                print(f'Epoch {epoch} | batch {i} / {int(train_data_amount // batch_size)} | loss: {cum_loss / ((i + 1) * batch_size):4f} |',
                      f'est. AUC: {sum(aucs) / (i // 100) :4f} | time: {round(time() - t_s)} sec(s)', end='\r')
                
        print(f'Epoch {epoch} loss: {cum_loss / train_data_amount}, time: {(time() - t_s):4f} sec(s)')
    
        torch.save(model.state_dict(), results_dir / 'model' / f'model_{epoch}.pth')

        model.eval()
        cum_loss = 0.0
        targets, preds = [], []
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                target = batch['target'].cuda()
                input_keyword = batch['keyword'].cuda()
                input_title = batch['title'].cuda()
                input_description = batch['description'].cuda()
                input_query = batch['query'].cuda()
                input_advertiser = batch['advertiser'].cuda()
                input_ad = batch['ad'].cuda()
                input_display = batch['display'].cuda()
                input_num = batch['numeric'].cuda()
                output = model(input_keyword, input_title, input_description, input_query, input_advertiser, input_ad, input_display, input_num)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='sum')
                loss_value = loss.detach().item()
                cum_loss += loss_value
                targets.append(target.cpu().numpy())
                preds.append(output.cpu().numpy())
                if i % 10 == 0:
                    print(f'Epoch {epoch} | batch {i} / {int(test_data_amount // batch_size)} loss: {cum_loss / ((i + 1) * batch_size):4f}', end='\r')
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        auc = roc_auc_score(targets, preds)
        print(f'Epoch {epoch} test loss: {cum_loss / test_data_amount} test auc: {auc}, time: {(time() - t_s):4f} sec(s)')



if __name__ == '__main__':
    main()
