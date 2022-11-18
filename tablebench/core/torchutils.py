import torch


class OnDeviceDataLoader(torch.utils.data.DataLoader):
    """DataLoader that automatically sends elements to a device."""

    def __init__(self, device: str, **kwargs):
        self.device_ = torch.device(device)
        super().__init__(**kwargs)

    def __next__(self):
        data = super().__next__(self)
        for x in data:
            x.to(self.device_)
        return data
