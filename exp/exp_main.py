from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_elec
from exp.exp_basic import Exp_Basic
# from models import Informer, Autoformer, Transformer
from models import Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader
        
    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_elec,
            'ETTh2':Dataset_elec,
            'ETTm1':Dataset_elec,
            'ETTm2':Dataset_elec,
            'WTH':Dataset_elec,
            'TRAF':Dataset_elec,
            'EXCH':Dataset_elec,
            'ECL':Dataset_elec,
            'elec':Dataset_elec,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            #Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            data_set = args.data,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            # inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            # cols=args.cols
        )
        print(flag, len(data_set))
        
        for i in range(5):
            print(data_set[0][i].shape, end=' ')
            print(' ')
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(train_loader):
                #print(batch_x.shape)
                iter_count += 1
                model_optim.zero_grad()
                true, sample = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in)
                loss = criterion(sample, true)
                train_loss.append(loss.item())
                #print(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            updatehidden = early_stopping.updatehidden
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        #         iter_count += 1
        #         model_optim.zero_grad()
        #         batch_x = batch_x.float().to(self.device)

        #         batch_y = batch_y.float().to(self.device)
        #         batch_x_mark = batch_x_mark.float().to(self.device)
        #         batch_y_mark = batch_y_mark.float().to(self.device)

        #         # decoder input
        #         dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        #         dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        #         # encoder - decoder
        #         if self.args.use_amp:
        #             with torch.cuda.amp.autocast():
        #                 if self.args.output_attention:
        #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        #                 else:
        #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        #                 f_dim = -1 if self.args.features == 'MS' else 0
        #                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
        #                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        #                 loss = criterion(outputs, batch_y)
        #                 train_loss.append(loss.item())
        #         else:
        #             if self.args.output_attention:
        #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        #             else:
        #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        #             f_dim = -1 if self.args.features == 'MS' else 0
        #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
        #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        #             loss = criterion(outputs, batch_y)
        #             train_loss.append(loss.item())

        #         if (i + 1) % 100 == 0:
        #             print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        #             speed = (time.time() - time_now) / iter_count
        #             left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
        #             print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
        #             iter_count = 0
        #             time_now = time.time()

        #         if self.args.use_amp:
        #             scaler.scale(loss).backward()
        #             scaler.step(model_optim)
        #             scaler.update()
        #         else:
        #             loss.backward()
        #             model_optim.step()

        #     print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        #     train_loss = np.average(train_loss)
        #     vali_loss = self.vali(vali_data, vali_loader, criterion)
        #     test_loss = self.vali(test_data, test_loader, criterion)

        #     print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        #         epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        #     early_stopping(vali_loss, self.model, path)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break

        #     adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        # return

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()

    #             pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
    #             true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)

    #     return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     return

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(test_loader):
            true, sample = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
            preds.append(sample.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        preds = test_data.inverse_transform(preds)
        trues = test_data.inverse_transform(trues)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return mae, mse

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        preds = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(pred_loader):
            true, sample= self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = pred_data.inverse_transform(preds)
        preds = preds[...,-1].reshape(-1, preds.shape[-2])
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.savetxt(folder_path+'real_prediction.txt', preds, fmt='%.4f')
        return


    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    sample = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                sample= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.inverse:
                sample = dataset_object.inverse_transform(sample)
            
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            sample = sample[...,:,f_dim:]
            return batch_y, sample
