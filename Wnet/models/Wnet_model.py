import torch
from torch.nn import functional as F
from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from Wnet.models import lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import math
from tqdm import tqdm
import os.path as osp
from collections import OrderedDict
import torch.nn.functional as tf

# sobel_x = torch.tensor([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')

# sobel_y = torch.tensor([
#     [-1, -2, -1],
#     [0, 0, 0],
#     [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')




@MODEL_REGISTRY.register()
class Wnet(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        if opt.get('task'):
            self.task=opt.get('task',0)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer


    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineContinousAnnealingRestartCyclicLR':
            self.schedulers.append(lr_scheduler.CosineContinousAnnealingRestartCyclicLR(self.optimizer_g, **train_opt['scheduler_g']))
            self.schedulers.append(lr_scheduler.CosineContinousAnnealingRestartCyclicLR(self.optimizer_l, **train_opt['scheduler_l']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')


    def pre_process(self):
        # pad to multiplication of window_size
        padding_size = self.opt['padding_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % padding_size != 0:
            self.mod_pad_h = padding_size - h % padding_size
        if w % padding_size != 0:
            self.mod_pad_w = padding_size - w % padding_size
        self.lq = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)


        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
            temp_name={metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.dirname(val_data['turb_path'][0]).split('/')[-2]+'_'+osp.splitext(osp.basename(val_data['turb_path'][0]))[0]

            self.feed_data(val_data)

            self.pre_process()

            self.test()

            self.post_process()
            visuals = self.get_current_visuals()
            # if current_iter<100000:
            #     sr_img = tensor2img([visuals['detilt_result']])
            # else:
            #     sr_img = tensor2img([visuals['total_result']])

            sr_img = tensor2img([visuals['output']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()



            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    temp_name[name] = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += temp_name[name]

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_psnr_{temp_name["psnr"]}.png')
                imwrite(sr_img, save_img_path)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def feed_data(self, data):
        self.lq = data['turb'].to(self.device)
        self.tilt = data['tilt'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.scaler = torch.cuda.amp.GradScaler()
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None


        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        refine_optim_params = []
        trained_optim_params= []
        for k, v in self.net_g.named_parameters():
            # if k.split('.')[0]!='deblur_model' and k.split('.')[0]!='detilt_model':
            if k.split('.')[0]!='detilt_model':
                logger = get_root_logger()
                refine_optim_params.append(v)
                logger.warning(f'Params {k} will be quickly optimized.')
            else:
                logger = get_root_logger()
                trained_optim_params.append(v)
                logger.warning(f'Params {k} will be slowly or not optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, refine_optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        self.optimizer_l = self.get_optimizer(optim_type, trained_optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_l)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_l.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()
        with torch.cuda.amp.autocast():
            self.output = self.net_g(self.lq)
            l_pix=self.cri_pix(self.output,self.gt)
            l_perceptal=self.cri_perceptual(self.output,self.gt)
            loss_dict['l_pix'] = l_pix
            l_total += l_pix
            loss_dict['l_perceptual']=l_perceptal
            l_total +=l_perceptal
            loss_dict['l_total']=l_total


        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.step(self.optimizer_l)
        self.scaler.update()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['output']=self.output.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        return out_dict