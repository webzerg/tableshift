import torch


class OnDeviceDataLoader(torch.utils.data.DataLoader):
    """DataLoader that automatically sends elements to a device."""

    def __init__(self, device: str, **kwargs):
        self.device_ = torch.device(device)
        super().__init__(**kwargs)

    def _move_batch_to_device(self, x: torch.Tensor, y: torch.Tensor,
                              g: torch.Tensor):
        return (x.to(self.device_), y.to(self.device_), g.to(self.device_))

    def __iter__(self):
        batches = iter(self)
        for b in batches:
            yield self._move_batch_to_device(*b)
