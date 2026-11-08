"""
Kiến trúc Autoencoder bám sát Mục 5.3 (Trang 15) của bài báo gốc DCSS
(Vardakas et al., Machine Learning 2026, 115:81):
- Với dữ liệu ảnh 28x28 (Fashion-MNIST, EMNIST, K-MNIST):
    x_{28x28} -> Conv^5_32 -> Conv^5_64 -> Conv^3_128 -> Fc_m (tanh)
              -> Deconv^3_128 -> Deconv^5_64 -> Deconv^5_32 -> x_hat_{28x28}
  (stride = 2, kích hoạt ReLU ở các tầng ẩn, Tanh ở tầng nhúng Fc_m, khởi tạo He).
- Hỗ trợ nhánh Residual / Adaptive pathway trên không gian nhúng (z_tilde = z + Linear(z))
  theo mô tả DECRA tại Mục 2 (Trang 5) của bài báo DCSS và mô hình lai Consist-DCSS.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvAutoencoder28x28(nn.Module):
    """
    Convolutional-Deconvolutional Autoencoder chuẩn theo Mục 5.3 bài báo DCSS (Trang 15)
    cho ảnh 28x28 (784 chiều).
    """

    def __init__(self, latent_dim: int = 10) -> None:
        super().__init__()
        self.input_dim = 784
        self.latent_dim = latent_dim

        # Encoder: 28x28 -> 14x14 -> 7x7 -> 3x3 -> Fc_m (tanh)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(128 * 3 * 3, latent_dim)
        self.embed_act = nn.Tanh()

        # Decoder: Fc_m -> 3x3 -> 7x7 -> 14x14 -> 28x28
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 128 * 3 * 3),
            nn.ReLU(inplace=True),
        )
        self.deconv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

        # Nhánh Residual Adaptive cho DECRA (Trang 5 bài báo DCSS) và Consist-DCSS:
        # z_tilde = tanh(z + W_res * z + b_res)
        self.residual_adapter = nn.Linear(latent_dim, latent_dim, bias=True)
        self._init_weights()
        self.reset_affine_identity()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số theo phương pháp He (Kaiming) đúng mô tả Trang 15."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m is not self.residual_adapter:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc_enc.weight)
        nn.init.zeros_(self.fc_enc.bias)

    def reset_affine_identity(self, noise_std: float = 0.01) -> None:
        """Khởi tạo nhánh Residual gần 0 để ban đầu z_tilde xấp xỉ z."""
        with torch.no_grad():
            self.residual_adapter.weight.normal_(0.0, noise_std)
            self.residual_adapter.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        img = x.view(-1, 1, 28, 28)
        feat = self.conv_encoder(img).flatten(start_dim=1)
        return self.embed_act(self.fc_enc(feat))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 128, 3, 3)
        img_hat = self.deconv_decoder(h)
        return img_hat.flatten(start_dim=1)

    def forward_affine(self, z: torch.Tensor) -> torch.Tensor:
        """Nhánh Residual pathway: tinh chỉnh biểu diễn song song với z gốc."""
        return torch.tanh(z + self.residual_adapter(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class SharedAutoencoder(nn.Module):
    """
    Wrapper tự động chọn ConvAutoencoder28x28 (khi input_dim == 784 cho dữ liệu ảnh 28x28
    đúng Mục 5.3 bài báo DCSS) hoặc Fully-Connected AE với tầng nhúng Tanh cho dữ liệu bảng.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (500, 500, 2000),
        latent_dim: int = 10,
        use_conv_for_28x28: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_conv = use_conv_for_28x28 and (input_dim == 784)

        if self.is_conv:
            self.net = ConvAutoencoder28x28(latent_dim=latent_dim)
        else:
            enc_layers = []
            in_d = input_dim
            for h_d in hidden_dims:
                enc_layers.append(nn.Linear(in_d, h_d))
                enc_layers.append(nn.ReLU(inplace=True))
                in_d = h_d
            enc_layers.append(nn.Linear(in_d, latent_dim))
            enc_layers.append(nn.Tanh())
            self.encoder = nn.Sequential(*enc_layers)

            dec_layers = []
            in_d = latent_dim
            for h_d in reversed(hidden_dims):
                dec_layers.append(nn.Linear(in_d, h_d))
                dec_layers.append(nn.ReLU(inplace=True))
                in_d = h_d
            dec_layers.append(nn.Linear(in_d, input_dim))
            dec_layers.append(nn.Sigmoid())
            self.decoder = nn.Sequential(*dec_layers)

            self.residual_adapter = nn.Linear(latent_dim, latent_dim, bias=True)
            self.reset_affine_identity()

    def reset_affine_identity(self, noise_std: float = 0.01) -> None:
        if self.is_conv:
            self.net.reset_affine_identity(noise_std=noise_std)
        else:
            with torch.no_grad():
                self.residual_adapter.weight.normal_(0.0, noise_std)
                self.residual_adapter.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            return self.net.encode(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            return self.net.decode(z)
        return self.decoder(z)

    def forward_affine(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            return self.net.forward_affine(z)
        return torch.tanh(z + self.residual_adapter(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_conv:
            return self.net(x)
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat
