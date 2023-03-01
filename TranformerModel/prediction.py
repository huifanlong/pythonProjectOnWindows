from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch.nn as nn
import numpy as np
import math
import torch
import pandas as pd
import copy
import time


def event_converter(e):
    if e == "play":
        return 0
    elif e == "pause":
        return 1
    elif e == "ratechange":
        return 2
    elif e == "skip":
        return 3


def state_converter(s):
    if s == "playing":
        return 0
    elif s == "paused":
        return 1


def rate_converter(r):
    if r == 0.0:
        return 0
    elif r == 1:
        return 1
    elif r == 1.2:
        return 2
    elif r == 1.5:
        return 3
    elif r == 2.0:
        return 4


class CustomDataset(Dataset):
    def __init__(self, df_events, df_records, uv_index, transform=None, target_transform=None):
        # 此方法在构造dataset时会执行一次，用于从文件中读取数据
        self.df_events = df_events
        self.df_records = df_records
        self.uv_index = uv_index
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.uv_index)  # 实际长度以index为准

    def __getitem__(self, idx):
        events = self.df_events[self.df_events.index == self.uv_index[idx]].to_numpy().astype("int32")
        score = self.df_records[self.df_records.index == self.uv_index[idx]].to_numpy().astype("int32")
        label = 1 if score == 100 else 0  # 二分类label
        events = torch.from_numpy(events)
        label = torch.tensor(label, dtype=torch.int32)
        if self.transform:
            events = self.transform(events)
        if self.target_transform:
            label = self.target_transform(label)
        return events, label


def my_collate(batch):
    """自定义dataloader的分配函数，让其不将数据stack
    """
    targets = []
    text = []
    for sample in batch:
        # 对于点击流，都将其整理成长度为30的，方便进行batch处理
        ''' 如果不进行batch处理 则把if-else条件删除，直接保留最后一个条件的步骤'''
        if sample[0].size(0) < max_event_size:  # 长度不足30的进行补充[-1，-1，-1，-1]，这样的话四个属性分别进行embedding时的size都要+1
            text.append(
                torch.cat(
                    (sample[0], torch.from_numpy(np.zeros([max_event_size - sample[0].size(0), 4]).astype('int32'))),
                    dim=0))
        elif sample[0].size(0) > max_event_size:
            text.append(sample[0][:max_event_size, :])
        else:
            text.append(sample[0])  # 取得feature
        targets.append(sample[1])  # 取得label
    return text, targets


class Embeddings(nn.Module):
    def __init__(self, dim_lis, voc_lis):
        """d_model: 指词嵌入的维度, vocab: 指词表的大小."""
        super(Embeddings, self).__init__()
        self.emb_e = nn.Embedding(voc_lis[0], dim_lis[0])
        self.dim_e = dim_lis[0]
        self.emb_p = nn.Embedding(voc_lis[1], dim_lis[1])
        self.dim_p = dim_lis[1]
        self.emb_s = nn.Embedding(voc_lis[2], dim_lis[2])
        self.dim_s = dim_lis[2]
        self.emb_r = nn.Embedding(voc_lis[3], dim_lis[3])
        self.dim_r = dim_lis[3]
        self.batch_norm = nn.BatchNorm1d(max_event_size)

    def forward(self, x):
        """参数x: 点击流数据。如果没有batch是二维张量，有batch则是三位张量"""
        # 从最后一维截取，分别获得四个属性，分别进行嵌入。最后在进行拼接（拼接的轴是维度-1）
        # a = self.emb_e(torch.tensor([10], dtype=torch.int32))
        a = x[..., 0]
        data = torch.from_numpy(np.random.randint(0, 5, size=[64, 30]).astype('int32'))
        data = self.emb_e(data)
        y = self.emb_e(a)
        e = self.emb_e(x[..., 0]) * math.sqrt(self.dim_e)
        p = self.emb_p(x[..., 1]) * math.sqrt(self.dim_p)
        s = self.emb_s(x[..., 2]) * math.sqrt(self.dim_s)
        r = self.emb_r(x[..., 3]) * math.sqrt(self.dim_r)
        # out = self.batch_norm(torch.cat((e, p, s, r), dim=x.ndim - 1))
        return self.batch_norm(torch.cat((e, p, s, r), dim=x.ndim - 1))


class Vocabulary:
    def __init__(self, name):
        """初始化函数中参数name代表传入某种语言的名字"""
        self.name = name
        self.word2index = {}
        self.index2word = {0: -1}
        self.n_size = 1

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_size
            self.index2word[self.n_size] = word
            self.n_size += 1


def make_vocabularies():
    """从文件中读取数据"""
    df_events = pd.read_excel('mydata/processed_clickevent_edutec.xlsx', usecols=[1, 2, 3, 4, 6, 7],
                              converters={"event": event_converter, "state": state_converter,
                                          "rate": rate_converter}, index_col=[0, 1])
    df_records = pd.read_csv('../mydata/row/quiz_record_edutec.csv', header=None, usecols=[1, 2, 9],
                             names=['vid', 'id', 'score'],
                             index_col=[1, 0])
    uv_index = df_records.index.unique().to_numpy()  # 获取record中的index
    df_events = df_events[df_events.index.isin(uv_index)]  # 整理events：交集中的才留下
    uv_index = df_events.index.unique().to_numpy()  # 交集的index
    df_records = df_records[df_records.index.isin(uv_index)]  # 整理record：交集中的才留下

    """构建词典"""
    # event_vocab, position_vocab, state_vocab, rate_vocab = Vocabulary("event_vocab"), Vocabulary(
    #     "position_vocab"), Vocabulary("state_vocab"), Vocabulary("rate_vocab")

    event_vocab.addSentence(df_events.event.unique())
    position_vocab.addSentence(df_events.position.unique())
    state_vocab.addSentence(df_events.state.unique())
    rate_vocab.addSentence(df_events.rate.unique())

    """用词典的index替换原始点击流数据"""
    df_events.event.replace(event_vocab.word2index.keys(), event_vocab.word2index.values(), inplace=True)
    df_events.position.replace(position_vocab.word2index.keys(), position_vocab.word2index.values(), inplace=True)
    df_events.state.replace(state_vocab.word2index.keys(), state_vocab.word2index.values(), inplace=True)
    df_events.rate.replace(rate_vocab.word2index.keys(), rate_vocab.word2index.values(), inplace=True)

    return df_events, df_records, uv_index


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, dim_lis, voc_lis, label_size: int, d_model: int,
                 nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = Embeddings(dim_lis, voc_lis)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, label_size)
        # self.final_batch_norm = nn.BatchNorm1d(max_event_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.emb_e.weight.data.uniform_(-initrange, initrange)
        # self.encoder.emb_e.bias.mydata.zero_()
        self.encoder.emb_p.weight.data.uniform_(-initrange, initrange)
        # self.encoder.emb_p.bias.mydata.zero_()
        self.encoder.emb_s.weight.data.uniform_(-initrange, initrange)
        # self.encoder.emb_s.bias.mydata.zero_()
        self.encoder.emb_r.weight.data.uniform_(-initrange, initrange)
        # self.encoder.emb_r.bias.mydata.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # output = self.final_batch_norm(output)
        return output


class CopyTransformerModel(nn.Module):

    def __init__(self, voc_num, dim, label_size: int, d_model: int,
                 nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(voc_num, dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, label_size)
        # self.final_batch_norm = nn.BatchNorm1d(max_event_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=False):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src)
        src = self.pos_encoder(src)
        if src_mask:
            output = self.transformer_encoder(src, src_mask)
        else:
            output = self.transformer_encoder(src)
        output = self.decoder(output)
        # output = self.final_batch_norm(output)
        return output


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def plot(data, name):
    y = data  # loss值，即y轴
    x = range(len(data))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel(name)  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x, y, linewidth=1, linestyle="solid", label="train"+name)
    plt.legend()
    plt.title(name+' curve')
    plt.show()


def train(model: nn.Module, dataloader, seq_len, is_src_mask) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 1
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(seq_len)  # 这个mask可以不传给模型
    for batch, (records_batch, labels_batch) in enumerate(dataloader):
        input = torch.stack(records_batch)  # 二维Tensor的list转化为三维Tensor，->（batch_size,30,4）三维整数
        labels = torch.stack(labels_batch)  # 一维Tensor的list转化为二维Tensor
        # if input.size(0) != batch_size:
        #     src_mask = src_mask[:input.size(0), :input.size(0)]
        # embedded = embedding(input)  # ->Tensor(batch_size,30,16) 嵌入后的三维小数
        if is_src_mask:
            output = model(input, src_mask)  # 直接将输入和mask输入给模型，得到输出 ->Tensor(batch_size,max_event_length,label_size)
        else:
            output = model(input)
        # 三维输出考虑了每个点击事件的二分类结果，将其通过max转化为二维，然后计算loss。也许可以在全连接之前就转化？
        # _, output = torch.std_mean(output, dim=1, keepdim=False)
        output = torch.squeeze(output[:, seq_len-1:, :])  # 采用最后一个一个点击事件的输出作为最终输出
        predicts = torch.squeeze(torch.topk(output, k=1)[1])
        predicts2 = torch.argmax(output, dim=1)  # topk改成argmax也可
        correct_num = torch.eq(predicts, labels.type(torch.int64)).sum()
        loss = criterion(output, labels.type(torch.int64))

        a = L_SM(output)
        loss2 = NLLLoss(L_SM(output), labels.type(torch.int64))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            all_train_losses.append(cur_loss)
            all_train_acc.append(correct_num / len(labels))
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch + 1:5d}/{len(myDataset) // batch_size + 1:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} correct_rate {correct_num / len(labels):5.2f}')
            total_loss = 0
            start_time = time.time()


def copy_collate(batch):
    """自定义dataloader的分配函数，让其不将数据stack
    """
    targets = []
    text = []
    for sample in batch:
        text.append(sample[0])  # 取得feature
        targets.append(sample[1])  # 取得label
    return text, targets


class CopyTestDataset(Dataset):
    def __init__(self, voc_num, data_num, seq_len, transform=None, target_transform=None):
        # 此方法在构造dataset时会执行一次，用于从文件中读取数据
        self.data_num = data_num
        self.seq_len = seq_len
        self.transform = transform
        self.target_transform = target_transform

        self.data = torch.from_numpy(np.random.randint(0, voc_num, size=(data_num, seq_len)).astype('int64'))

    def __len__(self):
        return self.data_num  # 数据量长度

    def __getitem__(self, idx):
        a = self.data[idx, self.seq_len-1]
        return self.data[idx, :], self.data[idx, self.seq_len-1]

# if __name__ == '__main__':
batch_size = 8
max_event_size = 10  # 为了批处理，将点击流数据padding为30，30能够覆盖95%的原始数据
event_emb_size, position_emb_size, state_emb_size, rate_emb_size,  = 2, 10, 2, 2  # 定义属性的emb大小

total_emb_size_list, total_voc__size_list = [], []
total_emb_size_list.extend([event_emb_size, position_emb_size, state_emb_size, rate_emb_size])

# 为需要进行embedding的四个属性构造词典
event_vocab, position_vocab, state_vocab, rate_vocab = Vocabulary("event_vocab"), Vocabulary(
    "position_vocab"), Vocabulary("state_vocab"), Vocabulary("rate_vocab")
df_events, df_records, uv_index = make_vocabularies()
total_voc__size_list.extend([event_vocab.n_size, position_vocab.n_size, state_vocab.n_size, rate_vocab.n_size])

# 构造数据集
myDataset = CustomDataset(df_events, df_records, uv_index)
train_dataloader = DataLoader(myDataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
label_size = 2  # 分类数目
emsize = 16  # 嵌入维度
d_hid = 32  # 前馈模型的神经元数
nlayers = 1  # encoder堆叠的层数
nhead = 1  # 多头注意力头数
model = TransformerModel(total_emb_size_list, total_voc__size_list, label_size=label_size, d_model=emsize, nhead=nhead, nlayers=nlayers, d_hid=d_hid)
L_SM = nn.LogSoftmax(dim=1)
NLLLoss = torch.nn.NLLLoss()  # Pytorch负对数似然损失函数
criterion = nn.CrossEntropyLoss()
lr = 5  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.999)

copy_test_dataset = CopyTestDataset(voc_num=10, data_num=10000, seq_len=10)
copy_test_dataloader = DataLoader(copy_test_dataset, batch_size=32, shuffle=True, collate_fn=copy_collate)
copy_test_model = CopyTransformerModel(voc_num=10, dim=2, label_size=10, d_model=2, nhead=nhead, nlayers=nlayers, d_hid=d_hid)

epochs = 5
all_train_losses = []
all_train_acc = []
# best_model = None
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model=copy_test_model, dataloader=copy_test_dataloader, seq_len=10, is_src_mask=False)
    # train(model=model, dataloader=train_dataloader, seq_len=max_event_size)
    # val_loss = evaluate(model, val_data)
    # val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ')
    # f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_model = copy.deepcopy(model)
    # scheduler.step()

plot(all_train_losses, "loss")
plot(all_train_acc, "acc")

torch.save(model, '../mydata/transformer.pth')
