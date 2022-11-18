import torch


class OnDeviceDataLoader(torch.utils.data.DataLoader):
    """DataLoader that automatically sends elements to a device."""

    def __init__(self, device: str, **kwargs):
        self.device_ = torch.device(device)
        super().__init__(**kwargs)

    def _move_batch_to_device(self, batch):
        return tuple([tens.to(self.device_) for tens in batch])

    def __iter__(self):
        batches = iter(self)
        for b in batches:
            yield self._move_batch_to_device(b)
