class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      输入形状
        - 3D 张量，形状为： ``(batch_size, field_size, embedding_size)``。
      输出形状
        - 2D 张量，形状为： ``(batch_size, featuremap_num)``，其中 ``featuremap_num = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` 如果 ``split_half=True``，否则为 ``sum(layer_size)``。
      参数
        - **field_size** : 正整数，特征组的数量。
        - **layer_size** : int 列表，每层的特征图数量。
        - **activation** : 应用于特征图上的激活函数名称。
        - **split_half** : 布尔值。如果设置为 False，每个隐藏层的一半特征图将连接到输出单元。
        - **seed** : 用作随机种子的 Python 整数。
      引用
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError("layer_size 必须是长度大于 1 的列表(tuple)")

        self.layer_size = layer_size
        self.field_nums = [field_size]  # 初始化特征组数量
        self.split_half = split_half
        self.activation = activation_layer(activation)  # 激活函数层
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()  # 初始化一维卷积层列表
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))  # 添加一维卷积层

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError("split_half=True 时除最后一层外的层大小必须是偶数")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.to(device)  # 将模型移至指定设备

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("输入维度 %d 非预期，期望为3维" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]  # 输入层
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0 交叉特征映射
            x = torch.einsum('bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            x = x.reshape(batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            x = self.conv1ds[i](x)  # 应用卷积层

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)  # 收集最终结果
            hidden_nn_layers.append(next_hidden)  # 更新隐藏层

        result = torch.cat(final_result, dim=1)  # 拼接最终结果
        result = torch.sum(result, -1)  # 在最后一个维度上求和

        return result
