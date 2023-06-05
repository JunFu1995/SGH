from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from importlib import import_module
from .config import Config

def buildModel(netFile, netClass='Model'):

    module = import_module('nets.' + netFile)

    if netFile in ['CNNIQA','JCSAN','Resnet50','DeepSRQ','DBCNN','C']:
        model = getattr(module, netClass)()
    elif netFile in ['HyperIQA', 'StyleIQA']:
        model = getattr(module, netClass)(16, 112, 224, 112, 56, 28, 14, 7)
    elif netFile == 'maniqa':
        # config file
        model_config = Config({
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.13,

            # optimization
            "batch_size": 8,
            "learning_rate": 1e-5,
            "weight_decay": 1e-5,
            "n_epoch": 300,
            "val_freq": 1,
            "T_max": 50,
            "eta_min": 0,
            "num_avg_val": 5,
            "crop_size": 224,
            "num_workers": 8,
        })

        model = getattr(module, netClass)(
            embed_dim=model_config.embed_dim,
            num_outputs=model_config.num_outputs,
            dim_mlp=model_config.dim_mlp,
            patch_size=model_config.patch_size,
            img_size=model_config.img_size,
            window_size=model_config.window_size,
            depths=model_config.depths,
            num_heads=model_config.num_heads,
            num_tab=model_config.num_tab,
            scale=model_config.scale,
        )
    else:
        raise RuntimeError('Invalid network name: {}.'.format(netFile))

    return model

