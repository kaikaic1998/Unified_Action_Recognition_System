from mmcv import Config
import torch
import torch.nn as nn

from pyskl.apis import inference_recognizer, init_recognizer

def main():

    device = torch.device('cuda')

    config = Config.fromfile('configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py')

    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    if config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    GCN_model = init_recognizer(config, '.cache/stgcnpp_ntu120_xset_hrnet.pth', device)
    # last layer:
    # 
    # (cls_head): GCNHead(
    #     (loss_cls): CrossEntropyLoss()
    #     (fc_cls): Linear(in_features=256, out_features=120, bias=True)
    #   )
    #
    # loss function is the cross entropy loss

    # True = not freeze parameters
    # False = freeze parameters
    for param in GCN_model.parameters():
        param.requires_grad = False 

    # print(GCN_model.cls_head)
    # print(GCN_model.cls_head.__dict__)
    # print(GCN_model.cls_head.in_c) # 256
    # print(GCN_model.cls_head.num_classes) # 120
    # print(GCN_model.cls_head.fc_cls) # Linear(in_features=256, out_features=120, bias=True)
    
    GCN_model.cls_head.fc_cls = nn.Linear(GCN_model.cls_head.in_c, 2)
    print(GCN_model.cls_head)
    print('\n', GCN_model.cls_head.fc_cls)

if __name__ == '__main__':
    main()
