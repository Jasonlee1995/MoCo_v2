import torch, torchvision
import torch.nn as nn



class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=None, model_type='query'):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        
        self.load_checkpoint(pretrained, model_type)
        self.freeze_initialize()

    def freeze_initialize(self):
        for name, param in self.model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        
        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()

    def load_checkpoint(self, checkpoint_dir, model_type):
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        if model_type == 'query':
            self.model.load_state_dict(checkpoint['q_state_dict'], strict=False)
        elif model_type == 'key':
            self.model.load_state_dict(checkpoint['k_state_dict'], strict=False)

    def forward(self, img):
        return self.model(img)