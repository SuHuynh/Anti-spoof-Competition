import torch 
import torch.nn as nn
### Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self,  **kwargs):
        super(BAP, self).__init__()
    def forward(self,feature_maps,attention_maps):
        feature_shape = feature_maps.size() ## 12*768*26*26*
        attention_shape = attention_maps.size() ## 12*num_parts*26*26
        # print('attention shape: ', attention_shape)
        # print('feature shape: ', feature_shape)
        # print(feature_shape,attention_shape)
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps)) ## 12*32*768
        phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))
        phi_I = phi_I.view(feature_shape[0],-1)
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1) ##12*(32*768)
        pooling_features = raw_features*100
        # print(pooling_features.shape)
        return raw_features,pooling_features