# src/hyper_network.py

import torch
from torch import nn
from copy import deepcopy


class HyperNetwork(nn.Module):
    def __init__(self, parent_model, child_model, mode='first'):
        super(HyperNetwork, self).__init__()
        self.mode = mode
        if self.mode == 'first':
            self.child_num, self.child_shape = self.first_layer_info(child_model)
        elif self.mode == 'all':
            self.child_num, self.child_shape = self.model_info(child_model)
        elif self.mode == 'first_and_last':
            self.child_num, self.child_shape = self.first_and_last_layer_info(child_model)
        else:
            raise ValueError("Mode must be 'first' , 'first_and_last' or 'all'.")
        total = sum(self.child_num)
        self.child_model = child_model
        self.parent_model = nn.Sequential(parent_model, nn.Linear(list(parent_model.parameters())[-1].size(0), total))
        # print total number of parameters that the parent model needs to output
        print(f"Total number of parameters for the child model: {total}")

    @staticmethod
    def model_info(model):
        shape = [tuple(param.size()) for param in list(model.parameters())]
        num = [torch.prod(torch.tensor(param_shape)).item() for param_shape in shape]
        return num, shape

    @staticmethod
    def first_layer_info(model):
        first_param = next(model.parameters())
        shape = [tuple(first_param.size())]
        num = [torch.prod(torch.tensor(shape[0])).item()]
        return num, shape

    @staticmethod
    def first_and_last_layer_info(model):
        first_param = next(model.parameters())
        last_param = list(model.parameters())[-1]
        shape = [tuple(first_param.size()), tuple(last_param.size())]
        num = [torch.prod(torch.tensor(shape[0])).item(), torch.prod(torch.tensor(shape[1])).item()]
        return num, shape
        

    def _construct(self, child_params):
        """
        :param child_params: shape: (batch_size, total_num_of_params)
        :return: target models. list of models with length: batch_size
        """
        batch_size = child_params.size(0)
        children_models = [deepcopy(self.child_model) for _ in range(batch_size)]
        split_params = self._split(child_params)
        reshaped_params = self._reshape(split_params)
        if self.mode == 'all':
            for child_num, child in enumerate(children_models):
                for layer_num, params in enumerate(child.parameters()):
                    params.requires_grad_(False)
                    params.copy_(reshaped_params[layer_num][child_num])
        elif self.mode == 'first':
            for child_num, child in enumerate(children_models):
                # Get the first parameter only
                first_param = next(child.parameters())
                first_param.requires_grad_(False)
                first_param.copy_(reshaped_params[0][child_num])
        elif self.mode == 'first_and_last':
            for child_num, child in enumerate(children_models):
                # Get the first and last parameters only
                first_param = next(child.parameters())
                last_param = list(child.parameters())[-1]
                first_param.requires_grad_(False)
                last_param.requires_grad_(False)
                first_param.copy_(reshaped_params[0][child_num])
                last_param.copy_(reshaped_params[-1][child_num])
        else:
            raise ValueError("Mode must be 'first' or 'all'.")
        return children_models

    def _split(self, child_params):
        return list(torch.split(child_params, self.child_num, dim=1))

    def _reshape(self, split_params: list):
        for i, params in enumerate(split_params):
            split_params[i] = params.reshape(params.size(0), *self.child_shape[i])
        return split_params

    def _beforward(self, x):
        children_params = self.parent_model(x)
        children_models = self._construct(children_params)
        return children_models

    def forward(self, x, y):
        '''
        :param x: input to the parent model. shape: (batch_size, input_size)
        :param y: input to the child model. shape: (batch_size, input_size)
        :return: model output. shape: (batch_size, output_size)
        '''
        children_models = self._beforward(x) # for every sample of x -> a tailored model is created.
        children_output = [child_model(y) for child_model in children_models]
        return torch.vstack(children_output)

    def save_model(self, filepath):
        self.child_model.save_model(filepath + ".child.pt")
        self.parent_model[0].save_model(filepath + ".parent.pt")


class HierarchicalHyperNetwork(nn.Module):
    def __init__(self, models_list: list):
        """
        :param models_list: [parent_model, child_model, grandchild_model, ...]
        """
        super(HierarchicalHyperNetwork, self).__init__()
        self.models = models_list
        self.target_num, self.target_shape, self.total = [], [], []
        for i in range(1, len(self.models)):
            # from grandchild to parent since only the youngest sibling has the correct number of parameters.
            target_num, target_shape = self.model_info(self.models[-i])
            total = sum(target_num)
            self.models[-i-1] = nn.Sequential(self.models[-i-1], nn.Linear(list(self.models[-i-1].parameters())[-1].size(0), total))
            self.target_num.append(target_num)
            self.target_shape.append(target_shape)
            self.total.append(total)
        target_num, target_shape = self.model_info(self.models[0])
        total = sum(target_num)
        self.target_num.append(target_num)
        self.target_shape.append(target_shape)
        self.total.append(total)
        self.target_num.reverse()
        self.target_shape.reverse()
        self.total.reverse()

    @staticmethod
    def model_info(model):
        shape = [tuple(param.size()) for param in list(model.parameters())]
        num = [torch.prod(torch.tensor(param_shape)).item() for param_shape in shape]
        return num, shape

    def _construct(self, target_model, target_params):
        split_params = self._split(target_params)
        reshaped_params = self._reshape(split_params)
        for layer_num, params in enumerate(target_model.parameters()):
            params.requires_grad_(False)
            params.copy_(reshaped_params[layer_num][0])
            params.requires_grad_(True)
        return target_model

    def _split(self, target_params, target_num):
        return list(torch.split(target_params.squeeze(0), target_num, dim=0)) # turns a tuple into a list such that target_params can be modified later.

    def _reshape(self, split_params: list, shape):
        for i, params in enumerate(split_params):
            split_params[i] = params.reshape(*shape[i])
        return split_params

    def forward(self, inps: list):
        '''
        :param list: list of tensor inputs to the parent model. [x, y, z...] each with shape (batch_size, input_size)
        :return: last model output. shape: (batch_size, output_size)
        '''
        # init output as an empty tensor with the correct shape
        output = torch.empty(0, self.target_shape[-1][-1][-1])
        for i in range(len(inps[0])): # for every sample in the batch
            for j in range(len(self.models) - 1): # for every model in the hierarchy
                target_params = self.models[j](inps[j][i].unsqueeze(0))
                target_params = self._split(target_params, self.target_num[j+1])
                target_params = self._reshape(target_params, self.target_shape[j+1])
                for layer_num, params in enumerate(self.models[j+1].parameters()):
                    # copy the parameters of the target model to the model in the hierarchy
                    params.data += (-params.data + target_params[layer_num])
                    # params.requires_grad_(False)
                    # params.copy_(target_params[layer_num])
                    # params.requires_grad_(True)
            # last model in the hierarchy
            output = torch.vstack((output, self.models[-1](inps[-1][i].unsqueeze(0))))
        return output


if __name__ == '__main__':
    in_f, out_f = 10, 2
    batch_size = 4
    models = [nn.Linear(in_f, out_f, bias=True) for _ in range(3)]
    hnn = HierarchicalHyperNetwork(models)
    x = torch.rand(batch_size, in_f)
    y = torch.rand(batch_size, in_f)
    z = torch.rand(batch_size, in_f)
    o = hnn([x, y, z])
    print(o.shape)

# end of file
