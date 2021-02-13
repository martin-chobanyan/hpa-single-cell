import albumentations as A
import numpy as np
import torch
from tqdm import tqdm


class RISE:
    """An implemenation of RISE (Randomized Input Sampling for Explanation of black box models)"""
    def __init__(self, model, chunk_size=32, progress=False, device='cpu'):
        """Initialization

        Parameters
        ----------
        model: torch.nn.Module
        chunk_size: int, optional
        progress: bool, optional
        device: str
        """
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.progress = progress

    @staticmethod
    def generate_small_masks(n_masks, wh_small):
        """Generate the random binary masks at a smaller dimension

        Parameters
        ----------
        n_masks: int
            The number of masks to generate
        wh_small: tuple
            The (small) width and height of the masks, respectively

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (n_masks, h_small, w_small)
        """
        w_small, h_small = wh_small
        small_masks = np.random.randint(0, 2, size=(n_masks, h_small, w_small))
        small_masks = small_masks.astype(np.float32)
        return small_masks

    @staticmethod
    def scale_and_crop_masks(masks, wh_scales, crop_size):
        """Scale the small masks to a larger dimension and crop out a target dimension

        Parameters
        ----------
        masks: numpy.ndarray
            The small masks with shape (n_masks, h_small, w_small)
        wh_scales: tuple
            The scaling factor for width and height, respectively
        crop_size: tuple
            The width and height of the target mask size, respsectively

        Returns
        -------
        numpy.ndarray
        """
        w, h = crop_size
        _, h_small, w_small = masks.shape
        w_scale, h_scale = wh_scales

        transform_fn = A.Compose([
            A.Resize((h_small + 1) * h_scale, (w_small + 1) * w_scale),
            A.RandomCrop(h, w),
        ])

        scaled_masks = []
        for m in masks:
            scaled_masks.append(transform_fn(image=m)['image'])
        return np.stack(scaled_masks)

    def __call__(self, img, label_id=None, n_masks=2048, wh_small=(10, 10)):
        """Main method

        Parameters
        ----------
        img: torch.Tensor
        label_id: int, optional
        n_masks: int, optional
        wh_small: tuple, optional

        Returns
        -------
        numpy.ndarray
        """
        *_, h, w = img.shape
        w_small, h_small = wh_small
        w_scale = int(w / w_small)
        h_scale = int(h / h_small)

        masks = self.generate_small_masks(n_masks, (w_small, h_small))
        masks = self.scale_and_crop_masks(masks, (w_scale, h_scale), crop_size=(w, h))

        # take the element-wise product of the image with the masks
        mask_tensor = torch.from_numpy(masks)
        mask_tensor = mask_tensor.unsqueeze(1)
        masked_imgs = mask_tensor * img.cpu()

        outputs = []
        with torch.no_grad():
            idx = np.arange(0, n_masks, self.chunk_size)
            if self.progress:
                idx = tqdm(idx, desc='Computing weights')
            for i in idx:
                chunk = masked_imgs[i:i + self.chunk_size].to(self.device)
                output = self.model(chunk)
                output = torch.softmax(output, dim=1)
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).cpu().numpy()

        if label_id is None:
            return outputs
        else:
            weights = outputs[:, label_id]
            weights = weights.reshape((-1, 1, 1))
            class_heatmap = (weights * masks).mean(axis=0)
            return class_heatmap
