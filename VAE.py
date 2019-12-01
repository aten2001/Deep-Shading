
# coding: utf-8

# In[10]:


import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2) 

    def forward(self, x, labels):
        N, C, H, W = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.cat((x.reshape(N, 64*6*6), labels), 1)


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_dim = 64*6*6
        self.linear = nn.Linear(self.input_dim + 10, self.input_dim)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(1, 1, 2, stride=1)


    def forward(self, x, labels):
        N, D = x.size()
        x = F.relu(self.deconv1(self.linear(x).reshape(N, 64, 6, 6)))
        return self.deconv3(F.relu(self.deconv2(x)))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = 10
        self.input_dim = 6*6*64
        self._enc_mu = torch.nn.Linear(self.input_dim + 10, self.latent_dim) ##
        self._enc_log_sigma = torch.nn.Linear(self.input_dim + 10, self.latent_dim) ##
        self._dec = nn.Linear(self.latent_dim, self.input_dim + 10)
        self._sig = nn.Sigmoid()
        self.randn = torch.randn(100000).cuda()

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(sigma.size()).cuda()
        #std_z = self.randn[torch.randint(0, 100000, sigma.size())]

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick
    
    def encode_mu_sigma(self, inputs):
        h_enc = self.encoder(inputs)
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        return mu, sigma
    
    def test_sample(self, mu, sigma):
        std_z = torch.randn(sigma.size()).cuda()
        z = mu + sigma * Variable(std_z, requires_grad=False).cuda()
        return self._sig(self.decoder(F.relu(self._dec(z))))

    def forward(self, state, labels):
        h_enc = self.encoder(state, labels)
        z = self._sample_latent(h_enc)
        return self._sig(self.decoder(F.relu(self._dec(z)), labels))


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    #print("mean size", mean_sq.size())
    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1.0, 1)


def generation_loss(original, generated):
    #generated = torch.where(generated > 0.0, generated, torch.ones_like(generated)*1e-7)
    #return -F.kl_div(generated, original, reduction='sum')
    return -torch.sum(original * torch.log(generated) + (1.0-original) * torch.log(1.0 - generated), 1)


# In[11]:


input_dim = 28 * 28
batch_size = 32
device = torch.device("cuda")

transform = transforms.Compose(
    [transforms.ToTensor()])
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

subset_indices = list(range(1000))
trainset = torch.utils.data.Subset(mnist, subset_indices)
print('Number of samples: ', len(trainset))
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)


encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)
vae.train()
print("Is cuda", next(encoder.parameters()).is_cuda) #True


optimizer = optim.Adam(vae.parameters(), lr=0.001)
l = None
for epoch in range(30):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = Variable(inputs).to(device), Variable(classes).to(device)
        N = inputs.size(0)
        labels = torch.zeros(N, 10).to(device)
        labels[range(N), classes]=1.0
        optimizer.zero_grad()
        outputs = vae(inputs, labels)
        l1 = latent_loss(vae.z_mean, vae.z_sigma)
        l2 = generation_loss(inputs.reshape(N, input_dim), outputs.reshape(N, input_dim))
        loss = torch.mean(l1 + l2)
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

plt.imshow(vae(inputs, labels).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)


# In[16]:


# conditional
optimizer = optim.Adam(vae.parameters(), lr=0.0005)
for epoch in range(30):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = Variable(inputs).to(device), Variable(classes).to(device)
        N = inputs.size(0)
        labels = torch.zeros(N, 10).to(device)
        labels[range(N), classes]=1.0
        optimizer.zero_grad()
        outputs = vae(inputs, labels)
        l1 = latent_loss(vae.z_mean, vae.z_sigma)
        l2 = generation_loss(inputs.reshape(N, input_dim), outputs.reshape(N, input_dim))
        loss = torch.mean(l1 + l2)
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

plt.imshow(vae(inputs, labels).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)


# In[26]:


plt.imshow(vae(inputs, labels).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)


# In[5]:


# vanilla

optimizer = optim.Adam(vae.parameters(), lr=0.0005)
for epoch in range(30):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = Variable(inputs).to(device), Variable(classes).to(device)
        N = inputs.size(0)
        optimizer.zero_grad()
        outputs = vae(inputs)
        l1 = latent_loss(vae.z_mean, vae.z_sigma)
        l2 = generation_loss(inputs.reshape(N, input_dim), outputs.reshape(N, input_dim))
        loss = torch.mean(l1 + l2)
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())
plt.imshow(vae(inputs).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)


# In[15]:


vae.eval()
mu, sigma = vae.encode_mu_sigma(inputs)
result = vae.test_sample(mu+0.3, sigma)
plt.imshow(result.data[0].cpu().numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)

