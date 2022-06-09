import utils
import torch, torchvision
import torch.nn as nn



class MoCo(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        '''
        dim : feature dimension (default: 128)
        K   : queue size; number of negative keys (default: 65536)
        m   : moco momentum of updating key encoder (default: 0.999)
        T   : softmax temperature (default: 0.07)
        '''
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = torchvision.models.resnet50(num_classes=dim)
        self.encoder_k = torchvision.models.resnet50(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # create the queue
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = utils.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # dequeue and enqueue : replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = utils.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = utils.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        '''
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        '''

        # compute query features
        q = self.encoder_q(img_q)  # (N, C)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # (N, C)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute positive, negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (N, 1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (N, K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (N, K+1)

        # apply temperature
        logits /= self.T

        # labels : positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels