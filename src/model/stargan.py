from torch import nn

class StarGAN(nn.Module):
    """_summary_
    StarGAN implementaition based on https://arxiv.org/abs/1912.01865
    """
    def __init__(
            self,
            Generator,
            Discriminator,
            MappingNetwork,
            StyleEncoder
            ):
        super().__init__()
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.MappingNetwork = MappingNetwork
        self.StyleEncoder = StyleEncoder

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters])
        trainable_parameters = sum(
            [p.numel for p in self.parameters if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f'\nAll model parameters: {all_parameters}'
        result_info = result_info + f'\nTrainable parameters: {trainable_parameters}'

        return result_info
