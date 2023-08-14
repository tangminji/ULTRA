import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch_dct as dct
from pytorch_wavelets import DWT1D, IDWT1D, DWT2D, IDWT2D
import pywt

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if args.model_type in ['ours','ours_cl']:
            self.fil_lst = self.fiter_lst_para(args.data_len, args.J, args.wvlname, args.mode)
            self.dwt1D = DWT1D(J=args.J, wave=args.wvlname, mode=args.mode)
            self.idwt1D = IDWT1D(wave=args.wvlname, mode=args.mode)
            self.wave = args.wvlname
            self.fea_dim = args.data_len
            self.lam = args.lam
            self.f_type = args.f_type
            # self.dwt2D = DWT2D(J=args.J, wave=args.wvlname, mode=args.mode)
            # self.idwt2D = IDWT2D(wave=args.wvlname, mode=args.mode)
        self.norm = True if args.model_type == 'ours_cl' else False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, filter=None, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            if filter != None:
                # TODO: to determine args.J. For CIFAR10s, out.shape=[batch_size, data_len], we pick 1D calculation
                # J = pywt.dwt_max_level(self.fea_dim, pywt.Wavelet(self.wave).dec_len)#for 1D
                # J = pywt.dwtn_max_level(out.shape, self.wave, axes=[-2, -1])#for 2D
                pass
            if filter == 'dwt':
                out += self.lam * self.filter_DWT(out, self.f_type)
            elif filter == 'dct':
                out += self.filter_DCT(out)

            logits = self.linear(out)
            # Feature for CL
            if self.norm:
                out = F.normalize(out, dim=1)

        return logits, out

    def filter_DWT(self, embeddings, f_type):
        #TODO: to determine args.J
        #J = pywt.dwtn_max_level(embeddings.shape, self.wave)
        yl, yh = self.dwt1D(embeddings[:,None,:])
        assert len(yh)+1 == len(self.fil_lst)

        yl_fil = F.adaptive_avg_pool1d(self.fil_lst[0][None, None], int(yl.shape[-1]))
        if f_type == 'enh':
            yl = yl * torch.sigmoid(yl_fil)
        elif f_type == 'enh_red':
            yl = yl * torch.tanh(yl_fil)
        for i in range(len(yh)):
            yh_fil = F.adaptive_avg_pool1d(self.fil_lst[i+1][None, None], int(yh[i].shape[-1]))
            if f_type == 'enh':
                yh[i] = yh[i] * torch.sigmoid(yh_fil)
            elif f_type == 'enh_red':
                yh[i] = yh[i] * torch.tanh(yh_fil)
        emb_filtered = self.idwt1D((yl, yh))
        return emb_filtered.squeeze()

    def fiter_lst_para(self, data_len, J, wvlname, mode='zero'):
        #data_len=224,J=[5,8],wvlname=['coif1','db8']
        fil_lst = []
        dec_len = pywt.Wavelet(wvlname).dec_len
        coeff_len = data_len
        for i in range(J):
            coeff_len = pywt.dwt_coeff_len(coeff_len, dec_len, mode)
            fil_lst.append(nn.Parameter(torch.ones(coeff_len)))
        # reorganization dimensions to match yl,yh
        fil_lst.insert(0, nn.Parameter(torch.ones(coeff_len)))
        return nn.ParameterList(fil_lst)

    def filter_DCT(self, embeddings):
        '''
        TODO: newly added
        :param embeddings:
        :return:
        '''
        seq_frequencies = dct.dct_2d(embeddings)
        # batch_size, num_channels, output_size[0], output_size[1]
        frq_filter = F.adaptive_avg_pool2d(self._frq_filter[None, None], int(embeddings.shape[-1]))
        # normalize filter values to [0, 1] for learnable filter
        frq_filter = torch.sigmoid(frq_filter)
        # element-wise multiplication of filter with each row (no expansion needed due to prior pooling)
        seq_filtered = seq_frequencies * frq_filter
        # perform inverse DCT
        emb_filtered = dct.idct_2d(seq_filtered)

        return emb_filtered


def ResNet18(args, num_classes=10):
    return ResNet(args, PreActBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(args, num_classes=10):
    return ResNet(args, BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(args, num_classes=10):
    return ResNet(args, Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(args, num_classes=10):
    return ResNet(args, Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(args, num_classes=10):
    return ResNet(args, Bottleneck, [3,8,36,3], num_classes=num_classes)

import torchvision
from torchvision.models import resnet50

class ULTRA(nn.Module):
    def __init__(self, args):
        super(ULTRA, self).__init__()
        if args.model_type in ['ours','ours_cl']:
            self.fil_lst = self.fiter_lst_para(args.data_len, args.J, args.wvlname, args.mode)
            self.dwt1D = DWT1D(J=args.J, wave=args.wvlname, mode=args.mode)
            self.idwt1D = IDWT1D(wave=args.wvlname, mode=args.mode)
            self.wave = args.wvlname
            self.fea_dim = args.data_len
            self.lam = args.lam
            self.f_type = args.f_type
    def forward(self, out, filter=None):
        if filter != None:
                # TODO: to determine args.J. For CIFAR10s, out.shape=[batch_size, data_len], we pick 1D calculation
                # J = pywt.dwt_max_level(self.fea_dim, pywt.Wavelet(self.wave).dec_len)#for 1D
                # print('J', J)
                # J = pywt.dwtn_max_level(out.shape, self.wave, axes=[-2, -1])#for 2D
                pass
        if filter == 'dwt':
            out += self.lam * self.filter_DWT(out, self.f_type)
        elif filter == 'dct':
            out += self.filter_DCT(out)
        return out
    def filter_DWT(self, embeddings, f_type):
        #TODO: to determine args.J
        #J = pywt.dwtn_max_level(embeddings.shape, self.wave)
        yl, yh = self.dwt1D(embeddings[:,None,:])
        assert len(yh)+1 == len(self.fil_lst)

        yl_fil = F.adaptive_avg_pool1d(self.fil_lst[0][None, None], int(yl.shape[-1]))
        if f_type == 'enh':
            yl = yl * torch.sigmoid(yl_fil)
        elif f_type == 'enh_red':
            yl = yl * torch.tanh(yl_fil)
        for i in range(len(yh)):
            yh_fil = F.adaptive_avg_pool1d(self.fil_lst[i+1][None, None], int(yh[i].shape[-1]))
            if f_type == 'enh':
                yh[i] = yh[i] * torch.sigmoid(yh_fil)
            elif f_type == 'enh_red':
                yh[i] = yh[i] * torch.tanh(yh_fil)
        emb_filtered = self.idwt1D((yl, yh))
        return emb_filtered.squeeze()

    def fiter_lst_para(self, data_len, J, wvlname, mode='zero'):
        #data_len=224,J=[5,8],wvlname=['coif1','db8']
        fil_lst = []
        dec_len = pywt.Wavelet(wvlname).dec_len
        coeff_len = data_len
        for i in range(J):
            coeff_len = pywt.dwt_coeff_len(coeff_len, dec_len, mode)
            fil_lst.append(nn.Parameter(torch.ones(coeff_len)))
        # reorganization dimensions to match yl,yh
        fil_lst.insert(0, nn.Parameter(torch.ones(coeff_len)))
        return nn.ParameterList(fil_lst)

    def filter_DCT(self, embeddings):
        '''
        TODO: newly added
        :param embeddings:
        :return:
        '''
        seq_frequencies = dct.dct_2d(embeddings)
        # batch_size, num_channels, output_size[0], output_size[1]
        frq_filter = F.adaptive_avg_pool2d(self._frq_filter[None, None], int(embeddings.shape[-1]))
        # normalize filter values to [0, 1] for learnable filter
        frq_filter = torch.sigmoid(frq_filter)
        # element-wise multiplication of filter with each row (no expansion needed due to prior pooling)
        seq_filtered = seq_frequencies * frq_filter
        # perform inverse DCT
        emb_filtered = dct.idct_2d(seq_filtered)

        return emb_filtered
    
class CustomResNet(nn.Module):
    def __init__(self, resnet, num_classes, func, norm=False):
        super(CustomResNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 保留 ResNet50 的卷积层部分
        self.func = func # 变换函数
        self.classifier = nn.Linear(resnet.fc.in_features, num_classes) # 将分类器修改为 14 分类
        self.norm = norm
    def forward(self, x, filter=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.func(x, filter) # 执行变换操作
        logits = self.classifier(x)
        # Feature for CL
        if self.norm:
            x = F.normalize(x, dim=1)
        return logits, x

def ResNet50(args, num_classes=10):
    resnet = resnet50(pretrained=True)
    norm = True if args.model_type == 'ours_cl' else False
    func = ULTRA(args)
    net = CustomResNet(resnet, num_classes, func,norm = norm)
    return net
# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())
if __name__ == '__main__':
    from main_ce1 import init_args
    args = init_args()
    x = torch.randn((2, 3, 224, 224))
    net = ResNet50(args, num_classes=14)
    y, fea = net(x, filter=args.filter)
    print(y.shape)
    print(fea.shape)

    from data import get_Clothing1M_train_and_val_loader
    train_loader, val_loader, test_loader = get_Clothing1M_train_and_val_loader(args)
    for data, target, index in train_loader:
        y, fea = net(data, filter=args.filter)
        break
    print(y.shape)
    print(fea.shape)
    # net = ResNet18(args)
    # y, fea = net(Variable(torch.randn(1,3,32,32)))
    # print(y.shape)
    # print(fea.shape)