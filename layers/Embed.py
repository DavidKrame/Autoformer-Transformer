# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm
# import math

# def compared_version(ver1, ver2):
#     """
#     :param ver1
#     :param ver2
#     :return: ver1< = >ver2 False/True
#     """
#     list1 = str(ver1).split(".")
#     list2 = str(ver2).split(".")
    
#     for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
#         if int(list1[i]) == int(list2[i]):
#             pass
#         elif int(list1[i]) < int(list2[i]):
#             return -1
#         else:
#             return 1
    
#     if len(list1) == len(list2):
#         return True
#     elif len(list1) < len(list2):
#         return False
#     else:
#         return True

# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x


# class FixedEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(FixedEmbedding, self).__init__()

#         w = torch.zeros(c_in, d_model).float()
#         w.require_grad = False

#         position = torch.arange(0, c_in).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         w[:, 0::2] = torch.sin(position * div_term)
#         w[:, 1::2] = torch.cos(position * div_term)

#         self.emb = nn.Embedding(c_in, d_model)
#         self.emb.weight = nn.Parameter(w, requires_grad=False)

#     def forward(self, x):
#         return self.emb(x).detach()


# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='h'):
#         super(TemporalEmbedding, self).__init__()

#         minute_size = 4
#         hour_size = 24
#         weekday_size = 7
#         day_size = 32
#         month_size = 13

#         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)

#     def forward(self, x):
#         x = x.long()

#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])

#         return hour_x + weekday_x + day_x + month_x + minute_x


# class TimeFeatureEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF', freq='h'):
#         super(TimeFeatureEmbedding, self).__init__()

#         freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
#         d_inp = freq_map[freq]
#         self.embed = nn.Linear(d_inp, d_model, bias=False)

#     def forward(self, x):
#         return self.embed(x)


# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
#         super(DataEmbedding, self).__init__()

#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, x_mark):
#         x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
#         return self.dropout(x)


# class DataEmbedding_wo_pos(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
#         super(DataEmbedding_wo_pos, self).__init__()

#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, x_mark):
#         x = self.value_embedding(x) + self.temporal_embedding(x_mark)
#         return self.dropout(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding1(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding1, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        if c_in == 1:
            self.c_in = c_in 
            self.query = nn.Linear(c_in + 1, c_in + 1)
            self.key = nn.Linear(c_in + 1, c_in + 1)
            self.value = nn.Linear(c_in + 1, c_in + 1)
            self.tokenConv = nn.Conv1d(in_channels = c_in + 1, out_channels=d_model, kernel_size=(3,), padding = (1,), padding_mode='circular')
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
        else:
            self.c_in = c_in
            self.query = nn.Linear(c_in, c_in)
            self.key = nn.Linear(c_in, c_in)
            # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=(3,), padding = (1,), padding_mode ='circular')
            print(d_model)
            self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,kernel_size=(3,), padding=padding, padding_mode='circular', bias=False)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
                    
    def forward(self, x):
        if self.c_in ==1:
            x = torch.cat([x, x[:,:,-1:]], dim=-1)
        print(x.shape)
        query = self.query(x)
        key = self.key(x)
        x1 =  torch.fft.rfft(query.contiguous())
        x2 = torch.fft.rfft(key.contiguous())
        x3 = torch.fft.irfft(x1 * torch.conj(x2),dim=-1)
        if self.c_in % 2 != 0 and self.c_in != 1:
            x3 = torch.cat([x3, x3[:,:,-1:]], dim=-1)
        x3 = torch.softmax(x3, dim=-1)*x
        x3 = x3 + x
        x = self.tokenConv(x3.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
            self.fc = nn.Linear(5*d_model, d_model)
        else:
            self.fc = nn.Linear(4*d_model, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        if hasattr(self, 'minute_embed'):
            out = torch.cat((minute_x, hour_x, weekday_x, day_x, month_x), dim=2)
        else:
            out = torch.cat((hour_x, weekday_x, day_x, month_x), dim=2)
        #print(out.shape)
        out = self.fc(out)
        #print(out.shape)
        return out

class TemporalEmbedding1(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding1, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='t'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)
def exp_moving_ave(theta, beta):
    #theta = np.array(theta).reshape((-1, 1))
    m, n, p= theta.shape
    #v = np.zeros((m, 1))
    for i in range(1, n):
        theta[:, i, :] = (beta * theta[:, i-1, :] + (1 - beta) * theta[:, i, :])
    for i in range(1, m):
        theta[:, i, :] /= (1 - beta**i)
    return theta

class seasonality_embedding(nn.Module):
    def __init__(self, c_in, d_model, freq = 't'):
        super(seasonality_embedding, self).__init__()
        self.embed = nn.Embedding(c_in, d_model)
        self.bate = nn.Linear()
    def forward(x, order):
        x = x

class DataEmbedding(nn.Module) :
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)