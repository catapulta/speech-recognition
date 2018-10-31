import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from warpctc_pytorch import CTCLoss
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder
import phoneme_list
import Levenshtein as L
import logging
from model import UtteranceModel
import pdb

torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='train.log', level=logging.DEBUG)


# data loader
class UtteranceDataset(Dataset):
    def __init__(self, data_path='../data/train.npy', label_path='./data/train_phonemes.npy', context=100, test=False):
        self.context = context
        self.test = test
        self.data = np.load(data_path, encoding='latin1')
        self.labels = np.load(label_path) + 1 if not test else None  # index labels from 1 to n_labels
        self.num_entries = len(self.data)
        # self.num_entries = int(len(self.data)*.001) if not 'test' in data_path else int(len(self.data)*.1)

    def __getitem__(self, i):
        data = self.data[i]
        data = torch.from_numpy(data)
        if self.test:
            return data
        else:
            labels = self.labels[i]
            labels = torch.from_numpy(labels)
            return data, labels

    def __len__(self):
        return self.num_entries


def collate(batch):
    '''
    Collate function. Transform a list of different length sequences into a batch. Passed as an argument to the DataLoader.
    seq_list: list with size batch_size. Each element is a tuple where the first element is the predictor 
    data and the second element is the label.
    output: data on format (batch_size, var_len_sequence)
    '''
    if len(batch[0]) == 2:
        utts, labels = zip(*batch)
        lens = [seq.size(0) for seq in utts]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        utts = [utts[i] for i in seq_order]
        labels = [labels[i] for i in seq_order]
        return utts, labels
    else:
        utts = batch
        lens = [seq.size(0) for seq in utts]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        utts = [utts[i] for i in seq_order]
        return utts


class Levenshtein:
    def __init__(self, charmap):
        self.label_map = [' '] + charmap  # add blank to first entry
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0,
            beam_width=100
        )

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target, feature_lengths):
        feature_lengths = torch.Tensor(feature_lengths)
        prediction = torch.transpose(prediction, 0, 1)
        prediction = prediction.cpu()
        probs = F.softmax(prediction, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)

        ls = 0.
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in target[i].numpy())
            # print("Pred: {}, True: {}".format(pred, true))
            ls += L.distance(pred, true)
        return ls


# model trainer
class LanguageModelTrainer:
    def __init__(self, model, loader, val_loader, test_loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.loader = loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9)
        self.criterion = CTCLoss()#size_average=True, length_average=False)
        self.criterion = self.criterion.cuda() if torch.cuda.is_available() else self.criterion
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=6)
        self.LD = Levenshtein(phoneme_list.PHONEME_MAP)
        self.best_rate = 1e10
        self.decoder = CTCBeamDecoder(labels=[' '] + phoneme_list.PHONEME_MAP, blank_id=0)

    def train(self):
        self.model.train()  # set to training mode
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            training_epoch_loss = 0
            for batch_num, (inputs, targets) in enumerate(self.loader):
                # # debug
                # # Save init values
                # old_state_dict = {}
                # for key in model.state_dict():
                #     old_state_dict[key] = model.state_dict()[key].clone()
                #
                # # Your training procedure
                # loss = self.train_batch(inputs, targets)
                #
                # # Save new params
                # new_state_dict = {}
                # for key in model.state_dict():
                #     new_state_dict[key] = model.state_dict()[key].clone()
                #
                # # Compare params
                # for key in old_state_dict:
                #     if (old_state_dict[key] == new_state_dict[key]).all():
                #         print('No diff in {}'.format(key))
                # print('Batch loss is ', float(loss))

                loss = self.train_batch(inputs, targets)
                epoch_loss += loss
                training_epoch_loss += loss
                # training print
                batch_print = 40
                if batch_num % batch_print == 0 and batch_num != 0:
                    self.print_training(batch_num, self.loader.batch_size, training_epoch_loss, batch_print)
                    training_epoch_loss = 0

            epoch_loss = epoch_loss / (batch_num + 1)
            self.epochs += 1
            self.scheduler.step(epoch_loss)
            print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                  % (self.epochs, self.max_epochs, epoch_loss))
            self.train_losses.append(epoch_loss)
            # log loss
            tLog.log_scalar('training_loss', epoch_loss, self.epochs)
            # log values and gradients of parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                tLog.log_histogram(tag, value.data.cpu().numpy(), self.epochs)
                tLog.log_histogram(tag+'/grad', value.grad.data.cpu().numpy(), self.epochs)

            # every 1 epochs, print validation statistics
            epochs_print = 1
            if self.epochs % epochs_print == 0 and not self.epochs == 0:
                with torch.no_grad():
                    t = "#########  Epoch {} #########".format(self.epochs)
                    print(t)
                    logging.info(t)
                    ls = 0
                    lens = 0
                    for j, (val_inputs, val_labels) in (enumerate(self.val_loader)):
                        idx = np.random.randint(0, len(val_inputs))
                        print('Pred', self.gen_batch(val_inputs[idx:idx + 1]))
                        print('Ground', ''.join([phoneme_list.PHONEME_MAP[o - 1] for o in val_labels[idx]]))
                        val_output, _, feature_lengths = self.model(val_inputs)
                        ls += self.LD.forward(val_output, val_labels, feature_lengths)
                        lens += len(val_inputs)
                    ls /= lens
                    t = "Validation LD {}:".format(ls)
                    print(t)
                    logging.info(t)
                    t = '--------------------------------------------'
                    print(t)
                    logging.info(t)
                    # log loss
                    vLog.log_scalar('LD', ls, self.epochs)
                    if self.best_rate > ls:
                        torch.save(model.state_dict(), "models/checkpoint.pt")
                        self.best_rate = ls

    def print_training(self, batch_num, batch_size, loss, batch_print):
        t = 'At {:.0f}% of epoch {}'.format(
            batch_num * batch_size / self.loader.dataset.num_entries * 100, self.epochs)
        print(t)
        logging.info(t)
        t = "Training loss : {}".format(loss / batch_print)
        print(t)
        logging.info(t)
        t = '--------------------------------------------'
        print(t)
        logging.info(t)

    def train_batch(self, inputs, targets):
        lens_tar = torch.Tensor([len(target) for target in targets])  # lens of all targets (sorted by loader)
        targets = torch.cat(targets)
        targets = targets.cuda() if torch.cuda.is_available() else targets
        outputs, _, lens_in = self.model(inputs)  # T x B x num_phonema, ignore hidden
        lens_in = torch.Tensor(lens_in)
        loss = self.criterion(outputs, targets.int().cpu(), lens_in.int().cpu(), lens_tar.int().cpu())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # print([i for i in model.cnn.modules()][1].__dict__['_parameters']['weight'][0])
        return float(loss)  # avoid autograd retention

    def test(self):
        preds = []
        for i, inputs in enumerate(self.test_loader):
            pred = self.gen_batch(inputs)
            preds += pred
        return preds

    def gen_batch(self, data_batch):
        scores, _, out_lengths = model(data_batch)
        out_lengths = torch.Tensor(out_lengths)
        scores = torch.transpose(scores, 0, 1)
        probs = F.softmax(scores, dim=2).data.cpu()
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=out_lengths)
        out_seq = []
        for i in range(output.size(0)):
            chrs = [phoneme_list.PHONEME_MAP[o.item() - 1] for o in output[i, 0, :out_seq_len[i, 0]]]
            out_seq.append("".join(chrs))
        return out_seq


def write_results(results):
    with open('predictions.csv', 'w') as f:
        f.write('Id,Predicted\n')
        for i, r in enumerate(results):
            f.write(','.join([str(i), r]))
            f.write('\n')


if __name__ == '__main__':
    import os.path
    import logger

    tLog, vLog = logger.Logger("./logs/train_pytorch"), logger.Logger("./logs/val_pytorch")

    NUM_EPOCHS = 20
    BATCH_SIZE = 64

    model = UtteranceModel(len(phoneme_list.PHONEME_MAP)+1, cnn_compression=2)

    ckpt_path = 'models/checkpoint.pt'
    if os.path.isfile(ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_dict)
        print('Checkpoint weights loaded.')

    utdst = UtteranceDataset(data_path='./data/wsj0_train.npy', label_path='./data/wsj0_train_merged_labels.npy')
    val_utdst = UtteranceDataset(data_path='./data/wsj0_dev.npy', label_path='./data/wsj0_dev_merged_labels.npy')
    test_utdst = UtteranceDataset('./data/wsj0_test.npy', test=True)
    loader = DataLoader(dataset=utdst, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=6)
    val_loader = DataLoader(dataset=val_utdst, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=6)
    test_loader = DataLoader(dataset=test_utdst, batch_size=1, shuffle=False, collate_fn=collate, num_workers=1)

    trainer = LanguageModelTrainer(model=model, loader=loader, val_loader=val_loader, test_loader=test_loader,
                                   max_epochs=NUM_EPOCHS, run_id='1')

    trainer.train()
    write_results(trainer.test())
