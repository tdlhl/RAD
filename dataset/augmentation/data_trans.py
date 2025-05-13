from .transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from .transforms.color_transforms import GammaTransform
from .transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from .transforms.resample_transforms import SimulateLowResolutionTransform
from .transforms.spatial_transforms import SpatialTransform, MirrorTransform
from .params import default_3D_augmentation_params as params
import torchvision

medklip_trans = torchvision.transforms.Compose([
        MirrorTransform(params.get("mirror_axes")),
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        BrightnessTransform(params.get("additive_brightness_mu"),params.get("additive_brightness_sigma"),True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                         p_per_channel=params.get("additive_brightness_p_per_channel")),
        ContrastAugmentationTransform(p_per_sample=0.15),
        GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),p_per_sample=params["p_gamma"])
    ])

kad_trans = torchvision.transforms.Compose([
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        BrightnessTransform(params.get("additive_brightness_mu"),params.get("additive_brightness_sigma"),True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                         p_per_channel=params.get("additive_brightness_p_per_channel")),
        ContrastAugmentationTransform(p_per_sample=0.15),
        GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),p_per_sample=params["p_gamma"])
    ])
