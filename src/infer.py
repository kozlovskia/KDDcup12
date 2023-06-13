from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import Model
from dataset import KDDcupDataset


def main():
    checkpoint_path = Path('../results/model_2023_06_13/model_2.pth')
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True, parents=True)
    batch_size = 2048
    data_amount = 5 * 1e6

    dataset = KDDcupDataset('../data/D5M_test_x.tsv', target=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    keyword_processor_args = {'sequence_len': 16, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 32}
    title_processor_args = {'sequence_len': 32, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 64}
    description_processor_args = {'sequence_len': 50, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 100}
    query_processor_args = {'sequence_len': 128, 'num_words': 10000, 'embedding_dim': 128, 'conv_out_size': 256}
    soft_ordering_args = {'input_dim': 3*(32+64+100+256)+3*16+11, 'output_dim': 1, 'in_num_kernels': 256, 'chunk_size': 16}
    other_embedding_dim = 16

    model = Model(keyword_processor_args, title_processor_args, description_processor_args, 
                  query_processor_args, soft_ordering_args, other_embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.cuda()
    model.eval()

    print('Number of parameters:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_keyword = batch['keyword'].cuda()
            input_title = batch['title'].cuda()
            input_description = batch['description'].cuda()
            input_query = batch['query'].cuda()
            input_advertiser = batch['advertiser'].cuda()
            input_ad = batch['ad'].cuda()
            input_display = batch['display'].cuda()
            input_num = batch['numeric'].cuda()
            output = model(input_keyword, input_title, input_description, input_query, input_advertiser, input_ad, input_display, input_num)
            preds.append(output.cpu().numpy())
            if i % 10 == 0:
                print(f'Batch {i} / {int(data_amount // batch_size)}',  end='\r')
        print("Done!")

    preds = pd.DataFrame(np.concatenate(preds), columns=['a_kozlowski'])
    preds.to_csv(results_dir / 'preds.csv', index=False)
    print(preds)


if __name__ == '__main__':
    main()
