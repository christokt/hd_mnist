# hd_mnist
Compares classification robustness to noise using an MLP and hyperdimensional vectors.

I run most of these experiments on my Apple Silicon machine, so I set my PyTorch device to 'mps'. If you wish to run this on cuda instead just change the device to 'cuda'.

```
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```
or
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
