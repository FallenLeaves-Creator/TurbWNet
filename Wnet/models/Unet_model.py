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
import torchvision.transforms as transforms

scharr_x = torch.tensor([
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')

scharr_y = torch.tensor([
    [-3, -10, -3],
    [0, 0, 0],
    [3, 10, 3]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')




@MODEL_REGISTRY.register()
class UnetModel(SRModel):

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
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineContinousAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineContinousAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
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

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.lq.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.lq.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.lq[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
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

            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.test()

            self.post_process()
            visuals = self.get_current_visuals()
            # if current_iter<100000:
            #     sr_img = tensor2img([visuals['detilt_result']])
            # else:
            #     sr_img = tensor2img([visuals['total_result']])
            if self.task=='detilt':
                sr_img = tensor2img([visuals['detilt_result']])
                metric_data['img'] = sr_img
            elif self.task=='deblur':
                sr_img = tensor2img([visuals['deblur_result']])
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
        if self.task=='detilt':
            bgr_img = data['tilt'].to(self.device)
            ####  grad_2_channels      ##########
            # gray_img= transforms.Grayscale()(bgr_img)
            # gray_img_x_grad=tf.conv2d(gray_img,scharr_x,padding='same')
            # gray_img_y_grad=tf.conv2d(gray_img,scharr_y,padding='same')
            # self.lq=torch.cat((gray_img_x_grad,gray_img_y_grad),dim=1)
            # self.lq=transforms.Grayscale()(bgr_img)
            self.lq=bgr_img
            self.tilt = data['tilt'].to(self.device)
            if 'tilt_field' in data.keys():
                self.tilt_field=data['tilt_field'].to(self.device)
        elif self.task=='deblur':
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


        if train_opt.get('tilt_field_opt'):
            self.cri_tilt_field = build_loss(train_opt['tilt_field_opt']).to(self.device)
        else:
            self.cri_tilt_field = None

        if train_opt.get('MS_SSIM_opt'):
            self.cri_MS_SSIM = build_loss(train_opt['MS_SSIM_opt']).to(self.device)
        else:
            self.cri_MS_SSIM = None

        if self.cri_pix is None and self.cri_tilt_field is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def flow_warp(self,
                x,
                flow,
                interpolation='bilinear',
                padding_mode='border',
                align_corners=True):
        """Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        """
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                            f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)  # n * h * w * 2
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()
        with torch.cuda.amp.autocast():
            self.output = self.net_g(self.lq)
            # pixel loss
            # tilt_map=self.output[:,3:5,...].permute(0,2,3,1)
            # output_image=self.output[:,0:3,...]
            if self.task=='detilt':
                tilt_map=self.output.permute(0,2,3,1)
                l_pix_1=self.cri_tilt_field(tilt_map,self.tilt_field)
                detilt_image=self.flow_warp(self.tilt, tilt_map)
                l_pix_2=self.cri_pix(detilt_image,self.gt)
                l_pix=l_pix_1+l_pix_2
                    # l_pix_1 = self.cri_pix(detilt_image,self.gt)
                    # tilted_gt=self.flow_warp(self.gt, -tilt_map)
                    # l_pix_2 = self.cri_pix(tilted_gt,self.tilt)
                    # l_pix=0.5*l_pix_1+0.5*l_pix_2
                loss_dict['tilt_field_pix_1'] = l_pix_1
                loss_dict['tilt_l_pix_2'] = l_pix_2
                loss_dict['tilt_l_pix'] = l_pix
            ######  grad diff map loss####################
            # l_pix=0
            # for i in range(3):
            #     detilt_x_grad=tf.conv2d(detilt_image[:,i,...].unsqueeze(dim=1),sobel_x)
            #     detilt_y_grad=tf.conv2d(detilt_image[:,i,...].unsqueeze(dim=1),sobel_y)
            #     gt_x_grad=tf.conv2d(self.gt[:,i,...].unsqueeze(dim=1),sobel_x)
            #     gt_y_grad=tf.conv2d(self.gt[:,i,...].unsqueeze(dim=1),sobel_y)
            #     l_pix+=0.5*self.cri_pix(detilt_x_grad,gt_x_grad)+0.5*self.cri_pix(detilt_y_grad,gt_y_grad)
            # l_pix=l_pix/3
            ###########
            elif self.task=='deblur':

                # fft_gt=torch.fft.fft2(self.tilt)
                # fft_b=torch.fft.fft2(self.lq)

                # pred_fft_psf = torch.complex(self.output[:,0:3,...], self.output[:,3:6,...])
                # pred_b=pred_fft_psf*fft_gt
                # pred_combined_tensor = torch.stack([pred_b.real, pred_b.imag], dim=-1)
                # fft_b_combined_tensor = torch.stack([fft_b.real, fft_b.imag], dim=-1)
                # l_pix_1=self.cri_pix(pred_combined_tensor,fft_b_combined_tensor)
                # pred_ffr_psf_conjugate=pred_fft_psf.conj()
                # pred_output=pred_ffr_psf_conjugate*fft_b/(pred_ffr_psf_conjugate*pred_fft_psf+1)
                # output_image=torch.fft.ifft2(pred_output).real
                # l_pix_2=self.cri_MS_SSIM(output_image,self.tilt)
                l_pix_1=self.cri_pix(self.output,self.tilt)
                loss_dict['blur_l_pix_1'] = l_pix_1
                l_pix = l_pix_1
                if self.cri_MS_SSIM:
                    l_pix_2=self.cri_MS_SSIM(self.output,self.tilt)
                    l_pix = l_pix_1+l_pix_2
                    loss_dict['blur_l_pix_2'] = l_pix_2
                loss_dict['blur_l_pix'] = l_pix
            else:
                l_pix=0

        # l_pix_blur=self.cri_pix(output_image,self.tilt)
        # loss_dict['blur_l_pix'] = l_pix_blur
            l_total += l_pix
        # if current_iter<100000:
        #     l_total += l_pix+l_pix_blur
        # else:
        #     total_image=torch.round(torch.nn.functional.grid_sample(output_image, tilt_map, mode='bilinear', padding_mode='border', align_corners=True))
        #     l_pix_all = self.cri_pix(total_image,self.gt)
        #     l_total += 0.5*(l_pix+l_pix_blur)+0.5*l_pix_all


        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # tilt_map=self.output[:,3:5,...].permute(0,2,3,1)
        # output_image=self.output[:,0:3,...]
        # detilt_image=self.flow_warp(self.tilt, tilt_map)
        # total_image=self.flow_warp(output_image, tilt_map)
        # out_dict['deblur_result'] = output_image.detach().cpu()
        if self.task=='detilt':
            tilt_map=self.output.permute(0,2,3,1)
            detilt_image=self.flow_warp(self.tilt, tilt_map)
            out_dict['detilt_result'] = detilt_image.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        elif self.task=='deblur':
            deblur_result=self.output
            out_dict['deblur_result'] = deblur_result.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.tilt.detach().cpu()
        # out_dict['total_result']  = total_image.detach().cpu()
        return out_dict