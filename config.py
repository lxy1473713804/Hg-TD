import os, json
import datetime
import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('LXY -- Flow imputation')
        # params of data storage
        parser.add_argument("--root", type=str, default="./data", help="absolute path of the dataset")
        parser.add_argument("--dataset", type=str, default="pems08", help="dir name of the flow dataset")
        parser.add_argument("--missing_type", type=str, default="BM", help="dir name of the missing_type")
        parser.add_argument("--missing_ratio", type=int, default=20, help="dir missing ratio of the dataset")
        # params of dataset
        parser.add_argument("--file_suffix", type=str, default='.npy', help="the filename suffix of dataset")
        # params of model training
        parser.add_argument("--start_epoch", type=int, default=1, help="start epoch")
        parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--norm", type=bool, default=True, help="if normalization")
        parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
        #params of model define
        parser.add_argument("--bysrank", type=int, default=80, help="bys model core size")
        parser.add_argument("--patch_size", type=int, default=144, help="size of the patch")
        parser.add_argument("--stride", type=int, default=72, help="size of the stride")
        parser.add_argument("--padding", type=int, default=36, help="size of the padding")
        parser.add_argument("--embed_dim", type=int, default=28, help="size of the embed_dim")
        parser.add_argument("--time_mask_ratio", type=float, default=0.1, help="number of the time_mask_ratio")
        parser.add_argument("--spatial_mask_ratio", type=float, default=0.1, help="number of the time_mask_ratio")
        # other
        parser.add_argument("--checkpoint", type=str, default='0', help="checkpoint to load pretrained models")
        parser.add_argument(
            "--sample_interval", type=int, default=1, help="epoch interval between sampling of images from model"
        )
        parser.add_argument(
                "--evaluation_interval", type=int, default=100, help="epoch interval between evaluation from model"
        )
        parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
        parser.add_argument("--time", type=str, default=self._time(), help="the run time")

        self.args = parser.parse_args([])

        os.makedirs(os.path.join(self.args.root, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(self.args.root, "args"), exist_ok=True)
        os.makedirs(os.path.join(self.args.root, "evaluation"), exist_ok=True)

    def parse(self, save_args=True):
        # print(self.args)
        if save_args:
            self._save_args()
        return self.args
    
    def _time(self):
        now = datetime.datetime.now()
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        hour = str(now.hour).zfill(2)
        minute = str(now.minute).zfill(2)
        date_str = year + month + day + hour + minute
        return date_str
    
    def _save_args(self):
        out_path = os.path.join(self.args.root, "args", 
                                f'{self.args.missing_type}{self.args.missing_ratio}-{self.args.time}.args')
        with open(out_path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    def load_params(self, param_file):
        with open(param_file, 'r') as f:
            params = json.load(f)
        return params

