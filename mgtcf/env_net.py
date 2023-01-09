import torch
from torch import nn

# self.wind_embed = nn.Linear(1,16)
#         self.intencity_class_embed = nn.Linear(6,16)
#         self.move_velocity_embed = nn.Linear(1,16)
#         self.month_embed = nn.Linear(12, 16)
#         self.long_embed = nn.Linear(12,16)
#         self.lat_embed = nn.Linear(6,16)
#         self.history_d_12_embed = nn.Linear(8,16)
#         self.history_d_24_embed = nn.Linear(8, 16)
#         self.history_i_24_embed = nn.Linear(4, 16)

class Env_net(nn.Module):
    def __init__(self):
        super(Env_net, self).__init__()

        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        self.data_embed['location_long'] = nn.Linear(12, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(6, embed_dim)
        self.data_embed['history_direction12'] = nn.Linear(8, embed_dim)
        self.data_embed['history_direction24'] = nn.Linear(8, embed_dim)
        self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim)

        self.GPH_embed = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(8,8)
        )

        env_f_in = len(self.data_embed)*16+8*8
        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_in,env_f_in//2),
            nn.ReLU(),
            nn.Linear(env_f_in//2, env_f_in // 2),
            nn.ReLU(),
            nn.Linear(env_f_in // 2, 32)
        )

        self.trajectory_fc = nn.Linear(32,8)
        self.intensity_fc = nn.Linear(32, 4)

        # self.init_weights()


    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in")
                m.bias.data.fill_(0.01)

        self.apply(init_kaiming)


    def forward(self,env_data,gph):
        '''

        :param env_data: b,x_len
        :param gph: b,1,h,w
        :return:
        '''
        batch = gph.shape[0]
        embed_list = []
        for key in self.data_embed:
            now_embed = self.data_embed[key](env_data[key])
            embed_list.append(now_embed)
        embed_list.append(self.GPH_embed(gph).reshape(batch,-1))
        embed = torch.cat(embed_list,dim=1)
#       batch,env_f_in
        feature = self.evn_extract(embed)
        classf_traj = self.trajectory_fc(feature)
        classf_inte = self.intensity_fc(feature)
        return feature,classf_traj,classf_inte





