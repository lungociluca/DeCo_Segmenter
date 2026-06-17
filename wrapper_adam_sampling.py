import torch
import copy

from src.diffusion.flow_matching.adam_sampling import AdamLMSampler
import config as local_config

class WrapperAdamLMSampler(AdamLMSampler):

    def __init__(self, order, scheduler, guidance_fn, num_steps, guidance, timeshift, save_maps):
        super().__init__(order=order, scheduler=scheduler, guidance_fn=guidance_fn, num_steps=num_steps, 
                         guidance=guidance, timeshift=timeshift, save_maps=save_maps)

    def _impl_sampling(self, net, noise, condition, uncondition, extra_dict=None):
        batch_size = noise.shape[0]
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        t_cur = torch.zeros([batch_size,]).to(noise.device, noise.dtype)
        for i  in range(self.num_steps-1, self.num_steps):
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            cfg_condition = cfg_condition.to(torch.float32)
            copy_extra_dict = copy.deepcopy(extra_dict)
            copy_extra_dict["save_maps"] = self.save_maps and i==self.num_steps-1
            attention_maps = net(cfg_x, cfg_t, cfg_condition, extra_dict=copy_extra_dict)
        return attention_maps, attention_maps