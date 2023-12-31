# hd_mnist
Compares classification robustness to noise using an MLP and hyperdimensional vectors.

To install the required dependencies run (there's not many):

```
pip install -r requirements.txt
```

I run most of these experiments on my Apple Silicon machine, so I set my PyTorch device to 'mps'. If you wish to run this on cuda instead just change the device to 'cuda'.

```
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```
or
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Since the hyperdimensional vectors take up a lot of memory, I ended up making a new variable hd_device that is set to 'cpu', for handling the high memory computations. This can be left unchanged.

```
hd_device = torch.device('cpu')
```
