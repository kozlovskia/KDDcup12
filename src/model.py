import numpy as np
import torch
import torch.nn as nn


class SoftOrdering1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim, in_num_kernels, chunk_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_num_kernels = in_num_kernels
        self.chunk_size = chunk_size
        self.dropout_rate = 0.2

        self.dense_upsample = nn.Sequential(nn.BatchNorm1d(self.input_dim),
                                             nn.Dropout(self.dropout_rate),
                                             nn.utils.weight_norm(nn.Linear(self.input_dim, self.in_num_kernels * self.chunk_size, bias=False)), 
                                             nn.CELU(),
                                             )
        
        self.conv_block1 = nn.Sequential(nn.BatchNorm1d(self.in_num_kernels),
                                          nn.utils.weight_norm(nn.Conv1d(self.in_num_kernels, 
                                                                        self.in_num_kernels * 2, 
                                                                        kernel_size=5, 
                                                                        stride=1, 
                                                                        padding=2, 
                                                                        bias=False)),
                                          nn.AdaptiveAvgPool1d(self.chunk_size // 2),
                                          nn.ReLU(),
                                          )
        self.conv_block2 = nn.Sequential(nn.BatchNorm1d(self.in_num_kernels * 2),
                                          nn.Dropout(self.dropout_rate),
                                          nn.utils.weight_norm(nn.Conv1d(self.in_num_kernels * 2,
                                                                        self.in_num_kernels * 2,
                                                                        kernel_size=3,
                                                                        stride=1,
                                                                        padding=1,
                                                                        bias=False)),
                                          nn.ReLU(),
                                          )
        self.conv_block3 = nn.Sequential(nn.BatchNorm1d(self.in_num_kernels * 2),
                                          nn.Dropout(self.dropout_rate),
                                          nn.utils.weight_norm(nn.Conv1d(self.in_num_kernels * 2,
                                                                        self.in_num_kernels * 2,
                                                                        kernel_size=3,
                                                                        stride=1,
                                                                        padding=1,
                                                                        bias=False)),
                                          nn.ReLU(),
                                          )
        self.conv_block4 = nn.Sequential(nn.BatchNorm1d(self.in_num_kernels * 2),
                                          nn.Dropout(self.dropout_rate),
                                          nn.utils.weight_norm(nn.Conv1d(self.in_num_kernels * 2,
                                                                        self.in_num_kernels * 2,
                                                                        kernel_size=5,
                                                                        stride=1,
                                                                        padding=2,
                                                                        groups=self.in_num_kernels * 2,
                                                                        bias=False)),
                                          )
        self.out_layers = nn.Sequential(nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
                                         nn.Flatten(),
                                         nn.Dropout(self.dropout_rate),
                                         nn.utils.weight_norm(nn.Linear(self.in_num_kernels * self.chunk_size // 2, self.output_dim, bias=False)),
                                         )

    def forward(self, x):
        x = self.dense_upsample(x)
        x = x.view(-1, self.in_num_kernels, self.chunk_size)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        xs = x
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x + xs
        x = nn.functional.relu(x)
        x = self.out_layers(x)
        return x
        

class TextProcessingCNN(nn.Module):
    def __init__(self, sequence_len, num_words, embedding_dim, conv_out_size):
        super().__init__()
        self.sequence_len = sequence_len
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.conv_out_size = conv_out_size

        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_dim, padding_idx=0)

        self.conv_list = nn.ModuleList([nn.Conv1d(self.embedding_dim, 
                                                  self.conv_out_size,
                                                  kernel_size=kernel_size, 
                                                  stride=2) 
                                        for kernel_size in [3, 5, 7]])
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x_conv_list = [nn.functional.relu(conv(x)) for conv in self.conv_list]
        x_pool_list = [nn.functional.max_pool1d(x_conv, x_conv.size(2)).squeeze(2) for x_conv in x_conv_list]
        x = torch.cat(x_pool_list, dim=1)
        return x


class Model(nn.Module):
    def __init__(self, text_processor_keyword_args,
                 text_processor_title_args,
                 text_processor_description_args,
                 text_processor_query_args,  
                 soft_ordering_args,
                 advertiser_embedding_dim):
        super().__init__()
        self.text_processor_keyword_args = text_processor_keyword_args
        self.text_processor_title_args = text_processor_title_args
        self.text_processor_description_args = text_processor_description_args
        self.text_processor_query_args = text_processor_query_args
        self.soft_ordering_args = soft_ordering_args
        self.advertiser_embedding_dim = advertiser_embedding_dim

        self.text_processor_keyword = TextProcessingCNN(**self.text_processor_keyword_args)
        self.text_processor_title = TextProcessingCNN(**self.text_processor_title_args)
        self.text_processor_description = TextProcessingCNN(**self.text_processor_description_args)
        self.text_processor_query = TextProcessingCNN(**self.text_processor_query_args)

        self.advertiser_embedding = nn.Embedding(14705 + 1, self.advertiser_embedding_dim, padding_idx=0)

        self.soft_ordering = SoftOrdering1DCNN(**self.soft_ordering_args)

    def forward(self, keyword_x, title_x, description_x, query_x, advertiser_x, num_x):
        keyword_x = self.text_processor_keyword(keyword_x)
        title_x = self.text_processor_title(title_x)
        description_x = self.text_processor_description(description_x)
        query_x = self.text_processor_query(query_x)
        advertiser_x = self.advertiser_embedding(advertiser_x).view(advertiser_x.size(0), -1)
        num_x = num_x.view(num_x.size(0), -1)
        x = torch.cat((keyword_x, title_x, description_x, query_x, advertiser_x, num_x), dim=1)
        x = self.soft_ordering(x)
        return x
