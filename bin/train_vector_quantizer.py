import torch
from torch.utils.data import DataLoader
import numpy as np
from src.solver import BaseSolver
from src.data_load import ArciticDataset, BnfCollate
from src.bnftocode import Quantizer
from src.optim import Optimizer
import torch_optimizer as optim
from src.util import human_format, feat_to_fig


class Solver(BaseSolver):
    """Customized Solver."""
    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.best_loss = np.inf
        self.optimizer_dict = ''
        self.tmp_step_val = 0

    def fetch_data(self, data):
        """Move data to device"""
        data = [i.to(self.device) for i in data]
        return data

    def load_data(self):
        """ Load data for training/validation/plotting."""
        train_dataset = ArciticDataset(
            meta_file=self.config.data.train_fid_list,
            arctic_ppg_dir = self.config.data.arctic_ppg_dir,
        )

        # dev_dataset = ArciticDataset(
        #     meta_file=self.config.data.dev_fid_list,
        #     arctic_ppg_dir = self.config.data.arctic_ppg_dir,
        # )

        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.paras.njobs,
            shuffle=True,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=BnfCollate(),
        )
        # self.dev_dataloader = DataLoader(
        #     dev_dataset,
        #     num_workers=self.paras.njobs,
        #     shuffle=False,
        #     batch_size=self.config.hparas.batch_size,
        #     pin_memory=False,
        #     drop_last=False,
        #     collate_fn=BnfCollate(),
        # )
        msg = "Have prepared training set and dev set."
        self.verbose(msg)
    
    def load_pretrained_params(self):
        pretrain_model_file = self.config.data.pretrain_model_file
        pretrain_ckpt = torch.load(
            pretrain_model_file, map_location=self.device
        )
        model_dict = self.model.state_dict()
        # print(model_dict.keys())
        # # 1. filter out unnecessrary keys
        # print(pretrain_ckpt['model'].keys())
        # # print(pretrain_ckpt['optimizer'])
        # print(pretrain_ckpt['global_step'])
        # print(pretrain_ckpt['loss'])
        
        
        # print('decoder.decoders.4.src_attn.linear_q.weight'.split(".", maxsplit=1))
        # for k,v in pretrain_ckpt.items():
        #     print(k['model'])
        #     print(k['model'][0].split(".", maxsplit=1))
        # pretrain_dict = {k.split(".", maxsplit=1)[1]: v 
        #                  for k, v in pretrain_ckpt.items() if "spk_embedding" not in k 
        #                     and "wav2ppg_model" not in k and "reduce_proj" not in k}
        # ----------
        pretrain_dict = pretrain_ckpt['model']
        self.best_loss = pretrain_ckpt['loss']
        self.step = pretrain_ckpt['global_step']
        self.tmp_step_val = self.step
        self.optimizer_dict = pretrain_ckpt['optimizer']
        assert len(pretrain_dict.keys()) == len(model_dict.keys())

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrain_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def set_model(self):
        """Setup model and optimizer"""
        # Model
        self.model = Quantizer(self.config["model"]).to(self.device)
        
        if "pretrain_model_file" in self.config.data:
            self.load_pretrained_params()

        # model_params = [{'params': self.model.spk_embedding.weight}]
        model_params = [{'params': self.model.parameters()}]
        args = self.config["model"]
        # Loss criterion
        # self.loss_criterion = MaskedMSELoss()
        # Optimizer
        self.optimizer = Optimizer(model_params, **self.config["hparas"])
        if "pretrain_model_file" in self.config.data:
            self.optimizer.load_opt_state_dict(self.optimizer_dict)
        # self.optimizer = optim.Lamb(model_params, lr=0.001,clamp_value=1)
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
    
    def exec(self):
        self.verbose("Total training steps {}.".format(
            human_format(self.max_step)))

        n_epochs = 0
        self.model.train()
        # Set as current time
        self.timer.set()
        
        while self.step < self.max_step:
            for data in self.train_dataloader:
                # Pre-step: updata lr_rate and do zero_grad
                # self.optimizer.zero_grad()
                # lr_rate = self.optimizer.pre_step(self.step)
                # total_loss = 0

                # data to device
                ppgs, in_lengths = self.fetch_data(data)
                self.timer.cnt("rd")
                loss, commit_loss = self.model(
                    xs=ppgs,
                    ilens=in_lengths,
                )

                self.timer.cnt("fw")

                # Back-prop
                grad_norm = self.backward(loss)
                self.step += 1
                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    # self.progress("Tr stat | Loss - {:.4f} | Mel - {:.4f} | Spk-loss - {:.4f} | Grad. Norm - {:.2f} | {}"
                    #               .format(loss.cpu().item(),mel_loss,spk_loss, grad_norm, self.timer.show()))
                    self.progress("Tr stat | T Loss - {:.4f} | Grad. Norm - {:.2f} | {}"
                                  .format(loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr': loss})

                # Validation
                if (self.step == 0) or (self.step % self.valid_step == 0) or (self.step==self.tmp_step_val + 1):
                    # self.validate()
                    self.save_checkpoint(f'loss_step_{self.step}.pth', 'loss', loss.cpu().item())
                    
                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()
    

    
    # def validate(self):
    #     self.model.eval()
    #     dev_loss = 0.0
    #     mseloss = torch.nn.MSELoss()
    #     for i, data in enumerate(self.dev_dataloader):
    #         self.progress('Valid step - {}/{}'.format(i+1, len(self.dev_dataloader)))
    #         # Fetch data
    #         # ppgs, lf0_uvs, mels, lengths = self.fetch_data(data)
    #         ppgs, mels, in_lengths, \
    #             out_lengths, spk_ids, prosody_vec, _, _ = self.fetch_data(data)

    #         with torch.no_grad():
    #             loss, commit_loss, after_outs, before_outs, ys, olens = self.model(
    #                 xs=ppgs,
    #                 ilens= in_lengths,
    #                 ys=mels,
    #                 olens=out_lengths,
    #                 spembs=spk_ids,
    #                 prosody_vec=mels,
    #             )
    #             # spkdvs = self.compute_secondary_models(before_outs)
    #             # spkd_loss = mseloss(spkdvs,spk_ids)
    #             # loss = loss + spkd_loss

    #             # loss = self.loss_criterion(mel_pred, mels, out_lengths)
    #             # loss, after_outs, before_outs, logits, ys, labels, olens
    #             # loss = l1_loss + l2_loss
    #             # loss = self.loss_criterion(mel_pred, mels, out_lengths)
    #             dev_loss += loss.cpu().item()

    #     dev_loss = dev_loss / (i + 1)
    #     self.save_checkpoint(f'step_{self.step}.pth', 'loss', dev_loss, show_msg=False)
    #     if dev_loss < self.best_loss:
    #         self.best_loss = dev_loss
    #         self.save_checkpoint(f'best_loss_step_{self.step}.pth', 'loss', dev_loss)
    #     self.write_log('loss', {'dv_loss': dev_loss})

    #     # Resume training
    #     self.model.train()

