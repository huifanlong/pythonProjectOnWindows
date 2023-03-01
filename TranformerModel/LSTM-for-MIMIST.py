import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings('ignore')
torch.manual_seed(1)

epoch = 2
batch_size = 64
time_step = 28  # 时间步数（图片高度）（因为每张图像为28*28，而每一个序列长度为1*28，所以总共是28个1*28）
input_size = 28  # 每步输入的长度（每行像素的个数）
lr = 0.01
download_mnist = True

num_classes = 10  # 总共有10类
hidden_size = 128  # 隐层大小
num_layers = 1

# MINIST
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title("MNIST:%i" % train_data.train_labels[0])
# plt.show()


train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

data = next(iter(train_loader))  # train_loader是迭代器
# print(mydata[0].shape)  # data的第一个元素为64个（1*28*28的图像）
# print(mydata[1].shape)  # data的第二个元素为64个标签
# print("mydata[0]", mydata[0])
# print("mydata[1]", mydata[1])
# print(np.array(mydata).shape)
# 每次迭代为64张图片由batch_size决定， 1为通道数（灰白图片）

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels.numpy()[:2000]


# print(test_x.shape)
# print(test_y.shape)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        # LSTM层
        self.rnn_layer = nn.LSTM(
            input_size=input_size,  # 每行的像素点个数
            hidden_size=hidden_size,
            num_layers=num_layers,  # 层数
            batch_first=True,  # input和output会以batch_size为第一维度
        )
        self.layer_size = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = time_step
        self.attention_size = time_step
        # self.w_omega = nn.Parameter(torch.zeros(self.hidden_size * self.layer_size, self.attention_size), requires_grad=True)
        # self.u_omega = nn.Parameter(torch.zeros(self.attention_size), requires_grad=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            self.hidden_size * self.layer_size, self.hidden_size * self.layer_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size * self.layer_size, 1))

        # 输出层
        self.linear_layer = nn.Linear(hidden_size, num_classes)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def attn(self, out_put):
        # Attention过程
        u = torch.tanh(torch.matmul(out_put, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = out_put * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        return feat

    def forward(self, x):
        # x.shape (batch, time_step, input_size)(64,28,28)
        # rnn_out.shape (batch, time_step, output_size)
        # h_n (n_layers, batch, hidden_size) LSTM有两个hidden states, h_c是分线， h_c是主线
        # c_n (n_layers, batch, hidden_size)
        rnn_output, (h_n, c_n) = self.rnn_layer(x, None)  # None表示hidden state 会用全0的state
        # 选择lstm_output[-1] 也就是最后一个输出，因为每个cell都会有输出，但我们只关心最后一个(分类问题)
        # 选取最后一个时间节点的rnn_output输出
        # 这里的 rnn_output[:, -1, :]的值也是h_n的值

        # non-attention
        # output = self.linear_layer(rnn_output[:, -1, :])

        # attention
        # attn_output = self.attention_net(rnn_output.permute(1, 0, 2))
        attn_output = self.attn(rnn_output)
        output = self.linear_layer(attn_output)

        return output


class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, num_class):
        super(BiLSTMAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)        #单词数，嵌入向量维度
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim*2]
        return context

    def forward(self, x):
        embedding = self.dropout(x)  # [seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        # output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

        attn_output = self.attention_net(output)
        logit = self.fc(attn_output)
        return logit


rnn = BiLSTMAttention(input_size, hidden_size, num_layers, num_classes)
# rnn = RNN(input_size, hidden_size, num_layers, num_classes)
# print(rnn)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# rnn.to(device) #将模型送入cuda
# print('devices = ', device)

# print(len(train_loader))
# 训练流程
total_losses = []
for i in range(epoch):
    rnn.train()
    total_batch = len(train_loader)
    run_loss = 0.0
    total_num, correct_num = 0, 0
    for step, (images, labels) in enumerate(train_loader):
        #         images, labels = images.to(device), labels.to(device)
        images = images.view(-1, 28, 28)

        # 运行模型
        outputs = rnn(images)
        # 损失函数
        loss = loss_func(outputs, labels)
        # 清除梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新
        optimizer.step()

        run_loss = loss.item()

        if (step + 1) % 100 == 0:
            print('Epoch[{}/{}], step[{}/{}], train_loss:{}'.format(i + 1, epoch, step + 1, total_batch,
                                                                    '%.4f' % run_loss))
            total_losses.append(run_loss)
    # 训练结束

    # 对模型进行测试
    with torch.no_grad():
        rnn.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            #             images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 28, 28)
            outputs = rnn(images)  # output(64, 10)  输出为（batch_size, output_size） 因为我们只关心最后一维输出
            # 输出是每一批64个样本，每个样本有10个概率值（对应十个分类） 将概率值最大数值的所在类作为当前的预测结果
            if i == 0:
                print("输出的结果", )
                print("输出的维度", outputs.shape)
            _, prediction = torch.max(outputs.data, 1)  # prediction保存最大值的索引,也就相当于标签数0-9
            if i == 0:
                print("最大值和索引", (_, prediction))
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        # 输出结果
        print(total, correct)
        print("Test accuracy of model in test images:{}".format(correct / total))

torch.save(rnn.state_dict(), 'rnn.pkl')
showPlot(total_losses)
