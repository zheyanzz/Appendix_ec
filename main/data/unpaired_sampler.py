

from torch.utils.data import DataLoader

from typing import Iterator


class UnpairedSampler:


    def __init__(self, content_loader: DataLoader, style_loader: DataLoader):

        self.content_loader = content_loader

        self.style_loader = style_loader


    def __iter__(self) -> Iterator:

        content_iter = iter(self.content_loader)

        style_iter = iter(self.style_loader)


        while True:

            try:

                content = next(content_iter)

            except StopIteration:

                content_iter = iter(self.content_loader)

                content = next(content_iter)


            try:

                style = next(style_iter)

            except StopIteration:

                style_iter = iter(self.style_loader)

                style = next(style_iter)


            yield content, style

