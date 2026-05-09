import torch
import torch.nn as nn
from Module.utils import MODEL_REGISTOR, MODEL_REGISTOR_MT, timer_wrap
from Module import *
from Module.modules import *
from layers import *
from CMT.transformer import TransformerEncoder

def get_model(model_name, input_shape, output_shape, *args, **kwargs):
    if model_name in MODEL_REGISTOR.registered_names():
        return MODEL_REGISTOR.get(model_name)(input_shape, output_shape, *args, **kwargs)
    else:
        raise NotImplementedError


def get_model_MT(model_name, *args, **kwargs):
    if model_name in MODEL_REGISTOR_MT.registered_names():
        return MODEL_REGISTOR_MT.get(model_name)(*args, **kwargs)
    else:
        raise NotImplementedError


class BaseModel(nn.Module):
    def __init__(self, input_shape=None, output_shape=None):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __build_pseudo_input(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        temp_x_ = torch.rand(input_shape)
        temp_x = temp_x_.unsqueeze(0)
        return temp_x

    def get_tensor_shape(self, forward_func, input_shape=None):
        pseudo_x = self.__build_pseudo_input(input_shape)
        pseudo_y = forward_func(pseudo_x)
        return pseudo_y.shape


@MODEL_REGISTOR.register()
class THHSCA_DREAMER(BaseModel):
########### 没有pooling
    def __init__(self, 
                 dropoutRate: float = 0.5, kernLength_1: int = 3, kernLength_2: int = 7, kernLength_3: int = 15):
        super().__init__()
        EEG_chan = 14
        PPS_chan = 2
        ###################
        EEG_F1 = EEG_chan * 2
        EEG_F2 = EEG_F1 * 2
        ##################
        PPS_F1 = PPS_chan * 2
        PPS_F2 = PPS_F1 * 2
        ##################
        downSample_1, downSample_2 = 4, 5

############# EEG #####################
        self.depth = [1, 2]
        self.stage0_1 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_1, groups=EEG_chan),
        )
        self.stage1_1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_1.append(
                nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_1, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_1 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_1.append(
                nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ####################################################################
        self.stage0_2 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_2, groups=EEG_chan),
        )
        self.stage1_2 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_2.append(
                nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_2, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_2.append(
                nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ###########################################################################
        self.stage0_3 = nn.Sequential(
            nn.Conv1d(EEG_chan, EEG_F2, kernLength_3, groups=EEG_chan),
        )
        self.stage1_3 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_3.append(
                nn.Sequential(
                    nn.Conv1d(EEG_F2, EEG_F2, kernLength_3, groups=EEG_F2),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_3 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_3.append(
                nn.Sequential(
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_3, padding=kernLength_3 // 2),
                    SeparableConv1d(EEG_F2, EEG_F2, kernel_size=kernLength_3, padding=kernLength_3 // 4),
                    nn.BatchNorm1d(EEG_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_3 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        #---------------------------------------------
##################################################################################
        self.CMT1 = TransformerEncoder(embed_dim=56,
                                       num_heads=8,
                                       layers=6,
                                       attn_dropout=0.1,
                                       relu_dropout=0.1,
                                       res_dropout=0.1,
                                       embed_dropout=0.1,
                                       attn_mask=False)
        self.CMT2 = TransformerEncoder(embed_dim=56,
                                       num_heads=8,
                                       layers=6,
                                       attn_dropout=0.1,
                                       relu_dropout=0.1,
                                       res_dropout=0.1,
                                       embed_dropout=0.1,
                                       attn_mask=False)


############### ECG ##################
        self.depth = [1, 2]
        self.stage0_ECG_1 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_1, groups=PPS_chan),
        )
        self.stage1_ECG_1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_ECG_1.append(
                nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_1, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_ECG_1 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_ECG_1.append(
                nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_1, padding=kernLength_1 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_ECG_1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##############################################################
        self.stage0_ECG_2 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_2, groups=PPS_chan),
        )
        self.stage1_ECG_2 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_ECG_2.append(
                nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_2, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_ECG_2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_ECG_2.append(
                nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_2, padding=kernLength_2 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_ECG_2 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        ##########################################################
        self.stage0_ECG_3 = nn.Sequential(
            nn.Conv1d(PPS_chan, PPS_F2, kernLength_3, groups=PPS_chan),
        )
        self.stage1_ECG_3 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1_ECG_3.append(
                nn.Sequential(
                    nn.Conv1d(PPS_F2, PPS_F2, kernLength_3, groups=PPS_F2),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2_ECG_3 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2_ECG_3.append(
                nn.Sequential(
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_3, padding=kernLength_3 // 2),
                    SeparableConv1d(PPS_F2, PPS_F2, kernel_size=kernLength_3, padding=kernLength_3 // 4),
                    nn.BatchNorm1d(PPS_F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1_ECG_3 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
######################################
        self.ECG_CMT1 = TransformerEncoder(embed_dim=8,
                                       num_heads=8, # 8
                                       layers=6,
                                       attn_dropout=0.1,
                                       relu_dropout=0.1,
                                       res_dropout=0.1,
                                       embed_dropout=0.1,
                                       attn_mask=False)
        self.ECG_CMT2 = TransformerEncoder(embed_dim=8,
                                       num_heads=8, # 8
                                       layers=6,
                                       attn_dropout=0.1,
                                       relu_dropout=0.1,
                                       res_dropout=0.1,
                                       embed_dropout=0.1,
                                       attn_mask=False)

        ############################################ EEG_GCN ###############################
        self.global_adj = nn.Parameter(torch.FloatTensor(56, 56), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.bn = nn.BatchNorm1d(56)
        self.bn_ = nn.BatchNorm1d(56)
        self.GCN = GraphConvolution(48, 48)
        ###############################################################################
        ############################################ ECG_GCN ###############################

        self.ECG_global_adj = nn.Parameter(torch.FloatTensor(8, 8), requires_grad=True)
        nn.init.xavier_uniform_(self.ECG_global_adj)
        self.ECG_bn = nn.BatchNorm1d(8)
        self.ECG_bn_ = nn.BatchNorm1d(8)
        self.ECG_GCN = GraphConvolution(48, 48)
        ###############################################################################

        self.layer1 = Chebynet(48, 2, 48)
        self.A = nn.Parameter(torch.FloatTensor(2, 2))
        nn.init.xavier_normal_(self.A)
        self.BN1 = nn.BatchNorm1d(2)


        self.classifier = nn.Sequential(
            nn.Linear(3168, 4)
        )

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        x = torch.squeeze(x,dim=1)
        # print(x.shape)
        eeg = x[:, 0:14, :]
        ecg = x[:, 14:16, :]

        eeg_embed = self.forward_embed(eeg)
        ecg_embed = self.forward_embed_ECG(ecg)

        eeg_GDF, ecg_GDF, inter = self.forward_GDF(eeg_embed, ecg_embed)
        eeg_GDF_, ecg_GDF_, inter_ = self.forward_GDF(eeg_GDF, ecg_GDF)
        inter_modal = inter + inter_

        out = torch.cat((eeg_GDF_, ecg_GDF_, inter_modal), dim=1)
        out = out.view(out.size()[0], -1)
        y2 = self.classifier(out)

        return y2
###############################################
    def forward_embed(self, x):

        x_brach_1 = self.stage0_1(x)
        for stage1_1 in self.stage1_1:
            x_brach_1 = stage1_1(x_brach_1)
        x_brach_1 = self.merge_s1_1(x_brach_1)
        for stage2_1 in self.stage2_1:
            x_brach_1 = stage2_1(x_brach_1)

        x_brach_2 = self.stage0_2(x)
        for stage1_2 in self.stage1_2:
            x_brach_2 = stage1_2(x_brach_2)
        x_brach_2 = self.merge_s1_2(x_brach_2)
        for stage2_2 in self.stage2_2:
            x_brach_2 = stage2_2(x_brach_2)


        x_brach_1 = x_brach_1.permute(2, 0, 1)
        x_brach_2 = x_brach_2.permute(2, 0, 1)
        x_fuse = torch.cat((x_brach_1, x_brach_2), dim=0)

        out1 = self.CMT1(x_brach_1, x_fuse, x_fuse)
        out2 = self.CMT2(x_brach_2, x_fuse, x_fuse)
        out = torch.cat((out1, out2), dim=0)
        out = out.permute(1, 2, 0)

        return out

    def forward_embed_ECG(self, x):
        x_brach_1 = self.stage0_ECG_1(x)
        for stage1_ECG_1 in self.stage1_ECG_1:
            x_brach_1 = stage1_ECG_1(x_brach_1)
        x_brach_1 = self.merge_s1_ECG_1(x_brach_1)
        for stage2_ECG_1 in self.stage2_ECG_1:
            x_brach_1 = stage2_ECG_1(x_brach_1)

        x_brach_2 = self.stage0_ECG_2(x)
        for stage1_ECG_2 in self.stage1_ECG_2:
            x_brach_2 = stage1_ECG_2(x_brach_2)
        x_brach_2 = self.merge_s1_ECG_2(x_brach_2)
        for stage2_ECG_2 in self.stage2_ECG_2:
            x_brach_2 = stage2_ECG_2(x_brach_2)


        x_brach_1 = x_brach_1.permute(2, 0, 1)
        x_brach_2 = x_brach_2.permute(2, 0, 1)
        x_fuse = torch.cat((x_brach_1, x_brach_2), dim=0)

        out1 = self.ECG_CMT1(x_brach_1, x_fuse, x_fuse)
        out2 = self.ECG_CMT2(x_brach_2, x_fuse, x_fuse)
        out = torch.cat((out1, out2), dim=0)
        out = out.permute(1, 2, 0)


        return out
##############################################
    def forward_GDF(self, eeg_embed, ecg_embed):

        eeg_adj = self.get_adj(eeg_embed)

        eeg_bn = self.bn(eeg_embed)
        eeg_gcn = self.GCN(eeg_bn, eeg_adj)
        eeg_bn_ = self.bn_(eeg_gcn)
        eeg_bn_ = eeg_bn_ + eeg_embed

        ecg_adj = self.get_ECG_adj(ecg_embed)
        ecg_bn = self.ECG_bn(ecg_embed)
        ecg_gcn = self.ECG_GCN(ecg_bn, ecg_adj)
        ecg_bn_ = self.ECG_bn_(ecg_gcn)
        ecg_bn_ = ecg_bn_ + ecg_embed

        eeg_agg = torch.unsqueeze(self.aggr_fun(eeg_bn_, 1), dim=1)
        ecg_agg = torch.unsqueeze(self.aggr_fun(ecg_bn_, 1), dim=1)

        feature_aggr = torch.cat((eeg_agg, ecg_agg), dim=1)

        feature_aggr = self.BN1(feature_aggr)
        L = normalize_A(self.A)
        result = self.layer1(feature_aggr, L)
        return eeg_bn_, ecg_bn_, result


    def get_adj(self, x, self_loop=True):
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def get_ECG_adj(self, x, self_loop=True):
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.ECG_global_adj + self.ECG_global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def get_EMG_adj(self, x, self_loop=True):
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.EMG_global_adj + self.EMG_global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def aggr_fun(self, x, dim):
        return torch.mean(x, dim=dim)


class GraphConvolution_dy(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):

        super(GraphConvolution_dy, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A: torch.Tensor, num_layers: int) -> torch.Tensor:
    support = []
    for i in range(num_layers):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvolution_dy(in_channels, out_channels))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L
