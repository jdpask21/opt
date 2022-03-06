import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LENGTH = 80   ###訓練データのシーケンスの数
BATCH_SIZE = 10   ###バッチサイズ
epochs = 1000
learning_rate = 1e-3

# テンソルの横方向の長さ
tensor_size = 4
# シーケンスの長さ
seq_length = SEQ_LENGTH
# 入力層のニューロン数(テンソルの横方向の長さ)
# LSTM/GRUに入力する特徴量をテンソルの横方向の長さ(`input_size`) * シーケンスの長さとする
input_size = tensor_size
# 隠れ層のニューロン層
hidden_size = 512
# 出力数(テンソルの横方向の長さ)
output_size = tensor_size




csv_file = open("./data_bit.csv", "r", encoding="ms932", errors="", newline="" )
#リスト形式
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

csv_list = []
for i in f:
    csv_list.append(i)


'''
#辞書形式
f = csv.DictReader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
'''

def list_to_tensor(cs, len_per_tensor=4):   ###CSV形式で受け取ったデータをテンソル型に変換する
    return_tensor = torch.randn(len(cs), len_per_tensor)
    #print(return_tensor)

    def del_quo(s):
        s = s.replace('\'', '')
        return s

    max_list = [0, 0, 0, 0]
    for index, row in enumerate(cs):
        if index == 0:
            continue
        #rowはList
        #row[0]で必要な項目を取得することができる
        #print(row)
        #row_str_pre = list(map(del_quo, row))
        #print(row_str_pre)
        row_float = list(map(float, row))
        tensor = torch.tensor(row_float[1:5])
        #print(type(row_float[1:5]))
        for i_ind, i in enumerate(row_float[1:5]):
            if max_list[i_ind] < i:
                #print(i_ind)
                max_list[i_ind] = i
        return_tensor[index] = tensor
        #print(tensor)
    return return_tensor, torch.tensor(max_list)

data, max_data = list_to_tensor(csv_list)

def make_training_dataset(get_data, max_tensor, len_per_tensor=4):
    if len(get_data) % SEQ_LENGTH == 0:
        dataset_tensor = torch.randn(int((len(get_data) - 1) / SEQ_LENGTH), SEQ_LENGTH, len_per_tensor)
        correct_tensor = torch.randn(int((len(get_data) - 1) / SEQ_LENGTH), len_per_tensor)
        max_seq_length = (len(get_data) - 1) / SEQ_LENGTH
    else:
        dataset_tensor = torch.randn(int(len(get_data) / SEQ_LENGTH), SEQ_LENGTH, len_per_tensor)
        correct_tensor = torch.randn(int(len(get_data) / SEQ_LENGTH), len_per_tensor)
        max_seq_length = len(get_data) / SEQ_LENGTH

    count_seq = 0
    dataset_index = 0
    for d_ind, d in enumerate(get_data):
        if count_seq < SEQ_LENGTH:
            dataset_tensor[dataset_index][count_seq] = d
            count_seq += 1
        else:   ###入力シーケンスが規定数に達した場合（SEQ_LENGTH）
            correct_tensor[dataset_index] = get_data[d_ind + 1] / max_tensor
            count_seq = 0   ###リセット
            dataset_index += 1   ###トレーニングデータの次のシーケンスに以降

        if dataset_index >= max_seq_length:   ###残りのデータ数がシーケンス数に満たない場合が終了条件
            break

    dataset_torch_util = torch.utils.data.TensorDataset(dataset_tensor, correct_tensor)
    return dataset_torch_util

def make_batch_dataset(dataset):
    # 各データセットのサンプル数を決定
    # train : val: test = 60% : 20% : 20%
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0)
    n_test = len(dataset) - n_train - n_val
    # ランダムに分割を行うため、シードを固定して再現性を確保
    torch.manual_seed(0)

    # データセットの分割
    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    # バッチサイズ
    batch_size = BATCH_SIZE
    # shuffle はデフォルトで False のため、学習データのみ True に指定
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size)
    return train_loader, val_loader, test_loader



dataset_tensor_ = make_training_dataset(data, max_data)
train_dataloader, val_dataloader, test_dataloader = make_batch_dataset(dataset_tensor_)

# モデルの構築
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM層にはinput_sizeにはimg_size、hidden_sizeはハイパーパラメータ、batch_firstは(batch_size, seq_length, input_size)を受け付けたいのでTrueにする
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # 全結合層のinputはLSTM層のoutput(batch_size, seq_length, hidden_size)と合わせる。outputはimg_size
        '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLu(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, tensor_size),
            nn.ReLU()
        )
        '''
        self.fc = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, tensor_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # y_rnnは(batch_size, seq_length, hidden_size)となる
        y_rnn, (h,c) = self.rnn(x, None)
        # yにはy_rnnのseq_length方向の最後の値を入れる
        y = self.fc(y_rnn[:, -1, :])
        y = self.sigmoid(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.fc3(y)
        y = self.sigmoid(y)
        #y = self.fc2(y)
        #y = self.fc3(y)
        return y

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for  batch, (X, y) in enumerate(dataloader):
            # 予測と損失の計算
            X = X.cuda()
            y = y.cuda()
            pred = lstm(X)
            pred = pred.cuda()
            loss = loss_fn(pred, y)

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()
            pred = model(X)
            pred = pred.cuda()
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

lstm = LSTM()
lstm.cuda()
print(lstm)


# loss functionの初期化、定義
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, lstm, loss_fn, optimizer)
    test_loop(test_dataloader, lstm, loss_fn)
print("Done!")
