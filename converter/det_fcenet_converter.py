import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

class DetFCENetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(DetFCENetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        # [print('paddle: {} ---- {}'.format(k, v.shape)) for k, v in para_state_dict.items()]
        # [print('pytorch: {} ---- {}'.format(k, v.shape)) for k, v in self.net.state_dict().items()]
        # exit()

        stages_replace_stage_flag = False
        for k, v in para_state_dict.items():
            if 'stage' in k:
                stages_replace_stage_flag = True
                break

        for k,v in self.net.state_dict().items():
            keyword = 'stages.'
            if keyword in k:
                if stages_replace_stage_flag is True:
                    # replace: 'stages.{}.' -> 'stage{}.'
                    name = k.replace('stages.', 'stage')
                else:
                    # replace: 'stages.{}.' -> ''
                    start_id = k.find(keyword)
                    end_id = start_id + len(keyword) + 1 + 1
                    name = k.replace(k[start_id:end_id], '')
            else:
                name = k

            name = name.replace('.lateral_convs_module.','.')
            name = name.replace('.fpn_convs_module.','.')

            if name.endswith('num_batches_tracked'):
                continue

            if name.endswith('running_mean'):
                ppname = name.replace('running_mean', '_mean')
            elif name.endswith('running_var'):
                ppname = name.replace('running_var', '_variance')
            elif name.endswith('bias') or name.endswith('weight'):
                ppname = name
            else:
                print('Redundance:')
                print(name)
                raise ValueError

            try:
                self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('exception:')
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    return res['Architecture']

if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)', default=None)
    parser.add_argument("--dst_model_path", type=str, help='save model path in pytorch', default=None)
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)
    else:
        cfg = {
            'model_type':'det',
            'algorithm':'FCE',
            'Transform':None,
            'Backbone':{
                'name': 'ResNet_vd',
                'layers': 50,
                'dcn_stage': [False, True, True, True],
                'out_indices': [1,2,3],
            },
            'Neck': {
                'name': 'FCEFPN',
                'out_channels': 256,
                'has_extra_convs': False,
                'extra_stage': 0,
                'in_channels': [512, 1024, 2048]
            },
            'Head': {
                'name': 'FCEHead',
                'fourier_degree': 5,
            }
        }

    # cfg = {'model_type':'det',
    #        'algorithm':'DB',
    #        'Transform':None,
    #        'Backbone':{'name':'ResNet', 'layers':18, 'disable_se':True},
    #        'Neck':{'name':'DBFPN', 'out_channels':256},
    #        'Head':{'name':'DBHead', 'k':50}}
    # kwargs = {'out_channels': 6625}
    kwargs = {}
    paddle_pretrained_model_path = os.path.join(os.path.abspath(args.src_model_path), 'best_accuracy')
    # paddle_pretrained_model_path = None
    converter = DetFCENetConverter(cfg, paddle_pretrained_model_path, **kwargs)
    print('todo')

    inp = torch.from_numpy(np.random.randn(1,3,736,1312).astype(np.float32))
    with torch.no_grad():
        out = converter.net(inp)

    # save
    if args.dst_model_path is not None:
        save_name = args.dst_model_path
    else:
        save_name = '{}infer.pth'.format(os.path.basename(os.path.dirname(paddle_pretrained_model_path))[:-5])
    converter.save_pytorch_weights(save_name)
    print('done.')