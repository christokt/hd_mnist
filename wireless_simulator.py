"""
Wireless Channel Simulator

A comprehensive wireless channel simulator for testing signal robustness
under various wireless channel conditions. Designed to integrate with
hyperdimensional computing and machine learning pipelines.

Supported Channel Models:
- AWGN (Additive White Gaussian Noise)
- Rayleigh Fading
- Rician Fading
- Multipath Channel
- Path Loss Models (Free Space, Log-Distance)
- Doppler Effects

Author: Claude
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum


class ChannelType(Enum):
    """Enumeration of supported wireless channel types."""
    AWGN = "awgn"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    MULTIPATH = "multipath"


class PathLossModel(Enum):
    """Enumeration of path loss models."""
    FREE_SPACE = "free_space"
    LOG_DISTANCE = "log_distance"
    HATA_URBAN = "hata_urban"


@dataclass
class ChannelConfig:
    """Configuration for wireless channel simulation."""
    snr_db: float = 20.0  # Signal-to-Noise Ratio in dB
    channel_type: ChannelType = ChannelType.AWGN
    k_factor: float = 3.0  # Rician K-factor (ratio of LOS to scattered power)
    num_taps: int = 4  # Number of multipath taps
    tap_delays: Optional[List[float]] = None  # Delay for each tap in samples
    tap_powers_db: Optional[List[float]] = None  # Power for each tap in dB
    doppler_freq: float = 0.0  # Maximum Doppler frequency in Hz
    sample_rate: float = 1e6  # Sample rate in Hz
    carrier_freq: float = 2.4e9  # Carrier frequency in Hz (default: 2.4 GHz)
    distance: float = 10.0  # Distance in meters for path loss
    path_loss_model: Optional[PathLossModel] = None
    path_loss_exponent: float = 2.0  # Path loss exponent for log-distance model
    normalize_output: bool = True  # Whether to normalize output power


class WirelessChannel:
    """
    Wireless channel simulator supporting various channel models.

    This class simulates the effects of wireless transmission on signals,
    including noise, fading, multipath propagation, and Doppler effects.

    Examples:
        >>> channel = WirelessChannel(snr_db=20, channel_type=ChannelType.AWGN)
        >>> received = channel(transmitted_signal)

        >>> channel = WirelessChannel(snr_db=15, channel_type=ChannelType.RAYLEIGH)
        >>> received = channel(transmitted_signal)
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        channel_type: Union[ChannelType, str] = ChannelType.AWGN,
        k_factor: float = 3.0,
        num_taps: int = 4,
        tap_delays: Optional[List[float]] = None,
        tap_powers_db: Optional[List[float]] = None,
        doppler_freq: float = 0.0,
        sample_rate: float = 1e6,
        carrier_freq: float = 2.4e9,
        distance: float = 10.0,
        path_loss_model: Optional[Union[PathLossModel, str]] = None,
        path_loss_exponent: float = 2.0,
        normalize_output: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the wireless channel simulator.

        Args:
            snr_db: Signal-to-Noise Ratio in decibels
            channel_type: Type of channel model (AWGN, RAYLEIGH, RICIAN, MULTIPATH)
            k_factor: Rician K-factor for RICIAN channel
            num_taps: Number of multipath taps for MULTIPATH channel
            tap_delays: Delay for each tap in samples (auto-generated if None)
            tap_powers_db: Power for each tap in dB (exponential decay if None)
            doppler_freq: Maximum Doppler frequency in Hz
            sample_rate: Sample rate in Hz
            carrier_freq: Carrier frequency in Hz
            distance: Distance in meters for path loss calculation
            path_loss_model: Path loss model to apply (None = no path loss)
            path_loss_exponent: Exponent for log-distance path loss model
            normalize_output: Whether to normalize output power
            device: PyTorch device for computation
            seed: Random seed for reproducibility
        """
        # Convert string to enum if needed
        if isinstance(channel_type, str):
            channel_type = ChannelType(channel_type.lower())
        if isinstance(path_loss_model, str):
            path_loss_model = PathLossModel(path_loss_model.lower())

        self.config = ChannelConfig(
            snr_db=snr_db,
            channel_type=channel_type,
            k_factor=k_factor,
            num_taps=num_taps,
            tap_delays=tap_delays,
            tap_powers_db=tap_powers_db,
            doppler_freq=doppler_freq,
            sample_rate=sample_rate,
            carrier_freq=carrier_freq,
            distance=distance,
            path_loss_model=path_loss_model,
            path_loss_exponent=path_loss_exponent,
            normalize_output=normalize_output
        )

        self.device = device or torch.device('cpu')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize multipath tap configuration
        self._init_multipath_taps()

    def _init_multipath_taps(self):
        """Initialize multipath tap delays and powers."""
        if self.config.tap_delays is None:
            # Default: uniformly spaced taps
            self.tap_delays = torch.arange(self.config.num_taps, device=self.device).float()
        else:
            self.tap_delays = torch.tensor(self.config.tap_delays, device=self.device)

        if self.config.tap_powers_db is None:
            # Default: exponentially decaying power profile
            decay = torch.arange(self.config.num_taps, device=self.device).float()
            self.tap_powers_db = -3.0 * decay  # 3 dB decay per tap
        else:
            self.tap_powers_db = torch.tensor(self.config.tap_powers_db, device=self.device)

        # Convert to linear scale and normalize
        self.tap_powers = 10 ** (self.tap_powers_db / 10)
        self.tap_powers = self.tap_powers / self.tap_powers.sum()

    def __call__(
        self,
        signal: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply wireless channel effects to the input signal.

        Args:
            signal: Input signal tensor (any shape)
            snr_db: Override SNR for this call (uses config value if None)

        Returns:
            Signal after passing through the wireless channel
        """
        return self.forward(signal, snr_db)

    def forward(
        self,
        signal: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply wireless channel effects to the input signal.

        Args:
            signal: Input signal tensor (any shape)
            snr_db: Override SNR for this call (uses config value if None)

        Returns:
            Signal after passing through the wireless channel
        """
        snr_db = snr_db if snr_db is not None else self.config.snr_db
        original_device = signal.device
        signal = signal.to(self.device)

        # Apply path loss if configured
        if self.config.path_loss_model is not None:
            signal = self._apply_path_loss(signal)

        # Apply channel-specific effects
        if self.config.channel_type == ChannelType.AWGN:
            output = self._apply_awgn(signal, snr_db)
        elif self.config.channel_type == ChannelType.RAYLEIGH:
            output = self._apply_rayleigh(signal, snr_db)
        elif self.config.channel_type == ChannelType.RICIAN:
            output = self._apply_rician(signal, snr_db)
        elif self.config.channel_type == ChannelType.MULTIPATH:
            output = self._apply_multipath(signal, snr_db)
        else:
            raise ValueError(f"Unknown channel type: {self.config.channel_type}")

        # Apply Doppler effect if configured
        if self.config.doppler_freq > 0:
            output = self._apply_doppler(output)

        # Normalize output if configured
        if self.config.normalize_output:
            output = self._normalize(output, signal)

        return output.to(original_device)

    def _apply_awgn(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply Additive White Gaussian Noise."""
        # Calculate signal power
        signal_power = torch.mean(signal ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise

    def _apply_rayleigh(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply Rayleigh fading channel."""
        # Generate complex Rayleigh fading coefficient
        # Rayleigh = magnitude of complex Gaussian
        h_real = torch.randn(1, device=self.device) / np.sqrt(2)
        h_imag = torch.randn(1, device=self.device) / np.sqrt(2)
        h_magnitude = torch.sqrt(h_real**2 + h_imag**2)

        # Apply fading
        faded_signal = signal * h_magnitude

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_rician(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply Rician fading channel."""
        k = self.config.k_factor

        # LOS component
        los_power = k / (k + 1)
        # Scattered component power
        scatter_power = 1 / (k + 1)

        # Generate Rician fading coefficient
        los_component = np.sqrt(los_power)
        scatter_real = torch.randn(1, device=self.device) * np.sqrt(scatter_power / 2)
        scatter_imag = torch.randn(1, device=self.device) * np.sqrt(scatter_power / 2)

        h_magnitude = torch.sqrt(
            (los_component + scatter_real)**2 + scatter_imag**2
        )

        # Apply fading
        faded_signal = signal * h_magnitude

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_multipath(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply multipath fading channel with multiple taps."""
        output = torch.zeros_like(signal)

        for i, (delay, power) in enumerate(zip(self.tap_delays, self.tap_powers)):
            # Generate Rayleigh fading for each tap
            h_real = torch.randn(1, device=self.device) / np.sqrt(2)
            h_imag = torch.randn(1, device=self.device) / np.sqrt(2)
            h_magnitude = torch.sqrt(h_real**2 + h_imag**2) * torch.sqrt(power)

            # Apply delayed and attenuated signal
            delay_int = int(delay.item())
            if delay_int == 0:
                output = output + signal * h_magnitude
            else:
                # Shift signal by delay
                if signal.dim() == 1:
                    shifted = torch.zeros_like(signal)
                    shifted[delay_int:] = signal[:-delay_int]
                    output = output + shifted * h_magnitude
                else:
                    # For multi-dimensional signals, apply delay to last dimension
                    shifted = torch.zeros_like(signal)
                    shifted[..., delay_int:] = signal[..., :-delay_int]
                    output = output + shifted * h_magnitude

        # Add AWGN
        return self._apply_awgn(output, snr_db)

    def _apply_doppler(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Doppler frequency shift effect."""
        fd = self.config.doppler_freq
        fs = self.config.sample_rate

        # Normalized Doppler frequency
        fd_norm = fd / fs

        # Generate time-varying phase
        if signal.dim() == 1:
            n = signal.shape[0]
        else:
            n = signal.shape[-1]

        t = torch.arange(n, device=self.device).float()

        # Random initial phase
        phi = torch.rand(1, device=self.device) * 2 * np.pi

        # Doppler-induced phase variation
        doppler_phase = 2 * np.pi * fd_norm * t + phi

        # Apply phase modulation (simplified model)
        doppler_factor = torch.cos(doppler_phase)

        if signal.dim() == 1:
            return signal * doppler_factor
        else:
            return signal * doppler_factor.view(*([1] * (signal.dim() - 1)), -1)

    def _apply_path_loss(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply path loss attenuation."""
        pl_db = self._calculate_path_loss()
        pl_linear = 10 ** (-pl_db / 20)  # Voltage ratio
        return signal * pl_linear

    def _calculate_path_loss(self) -> float:
        """Calculate path loss in dB based on the configured model."""
        d = self.config.distance
        fc = self.config.carrier_freq
        c = 3e8  # Speed of light
        wavelength = c / fc

        if self.config.path_loss_model == PathLossModel.FREE_SPACE:
            # Friis free space path loss
            if d > 0:
                pl = 20 * np.log10(4 * np.pi * d / wavelength)
            else:
                pl = 0

        elif self.config.path_loss_model == PathLossModel.LOG_DISTANCE:
            # Log-distance path loss model
            d0 = 1.0  # Reference distance
            n = self.config.path_loss_exponent
            if d > d0:
                pl_d0 = 20 * np.log10(4 * np.pi * d0 / wavelength)
                pl = pl_d0 + 10 * n * np.log10(d / d0)
            else:
                pl = 20 * np.log10(4 * np.pi * d / wavelength) if d > 0 else 0

        elif self.config.path_loss_model == PathLossModel.HATA_URBAN:
            # Simplified Hata model for urban environments
            # Valid for 150-1500 MHz, extended here for demonstration
            fc_mhz = fc / 1e6
            hb = 30  # Base station height in meters
            hm = 1.5  # Mobile height in meters

            # Antenna height correction factor for medium city
            ahm = (1.1 * np.log10(fc_mhz) - 0.7) * hm - (1.56 * np.log10(fc_mhz) - 0.8)

            # Path loss
            pl = (69.55 + 26.16 * np.log10(fc_mhz) - 13.82 * np.log10(hb) - ahm +
                  (44.9 - 6.55 * np.log10(hb)) * np.log10(d / 1000) if d > 0 else 0)
            pl = max(0, pl)

        else:
            pl = 0

        return pl

    def _normalize(self, output: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Normalize output to have similar power as input."""
        input_power = torch.mean(original ** 2)
        output_power = torch.mean(output ** 2)

        if output_power > 0:
            scale = torch.sqrt(input_power / output_power)
            return output * scale
        return output

    def get_channel_response(self, num_samples: int = 1000) -> torch.Tensor:
        """
        Get the channel impulse response.

        Args:
            num_samples: Number of samples for the response

        Returns:
            Channel impulse response tensor
        """
        impulse = torch.zeros(num_samples, device=self.device)
        impulse[0] = 1.0
        return self.forward(impulse, snr_db=100)  # High SNR to see channel response

    def ber_simulation(
        self,
        num_bits: int = 10000,
        snr_range_db: Optional[List[float]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Simulate Bit Error Rate (BER) for BPSK modulation.

        Args:
            num_bits: Number of bits to simulate
            snr_range_db: List of SNR values to test

        Returns:
            Tuple of (snr_values, ber_values)
        """
        if snr_range_db is None:
            snr_range_db = list(range(-5, 25, 2))

        ber_values = []

        for snr in snr_range_db:
            # Generate random bits
            bits = torch.randint(0, 2, (num_bits,), device=self.device).float()

            # BPSK modulation: 0 -> -1, 1 -> +1
            tx_signal = 2 * bits - 1

            # Pass through channel
            rx_signal = self.forward(tx_signal, snr_db=snr)

            # BPSK demodulation
            rx_bits = (rx_signal > 0).float()

            # Calculate BER
            errors = torch.sum(bits != rx_bits).item()
            ber = errors / num_bits
            ber_values.append(ber)

        return snr_range_db, ber_values


class ImageWirelessChannel(WirelessChannel):
    """
    Wireless channel simulator specialized for image transmission.

    This class extends WirelessChannel with image-specific features,
    making it easy to integrate with image classification pipelines.

    Examples:
        >>> channel = ImageWirelessChannel(snr_db=15)
        >>> noisy_image = channel(clean_image)

        >>> # Use with DataLoader transform
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     channel,
        ... ])
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        channel_type: Union[ChannelType, str] = ChannelType.AWGN,
        pixel_range: Tuple[float, float] = (0.0, 1.0),
        clip_output: bool = True,
        **kwargs
    ):
        """
        Initialize the image wireless channel.

        Args:
            snr_db: Signal-to-Noise Ratio in decibels
            channel_type: Type of channel model
            pixel_range: Expected range of pixel values (min, max)
            clip_output: Whether to clip output to pixel_range
            **kwargs: Additional arguments passed to WirelessChannel
        """
        super().__init__(snr_db=snr_db, channel_type=channel_type, **kwargs)
        self.pixel_range = pixel_range
        self.clip_output = clip_output

    def forward(
        self,
        image: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply wireless channel effects to an image.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            snr_db: Override SNR for this call

        Returns:
            Image after passing through the wireless channel
        """
        # Apply channel effects
        output = super().forward(image, snr_db)

        # Clip to valid pixel range
        if self.clip_output:
            output = torch.clamp(output, self.pixel_range[0], self.pixel_range[1])

        return output

    def apply_to_batch(
        self,
        images: torch.Tensor,
        independent_fading: bool = True
    ) -> torch.Tensor:
        """
        Apply channel effects to a batch of images.

        Args:
            images: Batch of images (B, C, H, W)
            independent_fading: If True, each image gets independent fading

        Returns:
            Batch of images after channel effects
        """
        if not independent_fading:
            return self.forward(images)

        # Apply independent fading to each image
        outputs = []
        for i in range(images.shape[0]):
            outputs.append(self.forward(images[i:i+1]))
        return torch.cat(outputs, dim=0)


class HDVectorChannel(WirelessChannel):
    """
    Wireless channel simulator specialized for hyperdimensional vectors.

    This class handles the transmission of HD vectors over wireless channels,
    with support for binary (±1) and real-valued HD vectors.

    Examples:
        >>> channel = HDVectorChannel(snr_db=10, hd_dimension=10000)
        >>> received_hv = channel(transmitted_hv)
    """

    def __init__(
        self,
        snr_db: float = 20.0,
        channel_type: Union[ChannelType, str] = ChannelType.AWGN,
        hd_dimension: int = 10000,
        binary_vectors: bool = True,
        hard_decision: bool = True,
        **kwargs
    ):
        """
        Initialize the HD vector wireless channel.

        Args:
            snr_db: Signal-to-Noise Ratio in decibels
            channel_type: Type of channel model
            hd_dimension: Dimension of hyperdimensional vectors
            binary_vectors: Whether vectors are binary (±1)
            hard_decision: Whether to apply hard decision at receiver
            **kwargs: Additional arguments passed to WirelessChannel
        """
        super().__init__(snr_db=snr_db, channel_type=channel_type, **kwargs)
        self.hd_dimension = hd_dimension
        self.binary_vectors = binary_vectors
        self.hard_decision = hard_decision

    def forward(
        self,
        hd_vector: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Transmit an HD vector through the wireless channel.

        Args:
            hd_vector: Input HD vector(s)
            snr_db: Override SNR for this call

        Returns:
            Received HD vector(s)
        """
        # Apply channel effects
        output = super().forward(hd_vector, snr_db)

        # Apply hard decision for binary vectors
        if self.binary_vectors and self.hard_decision:
            output = torch.sign(output)
            # Handle zeros (shouldn't happen often but just in case)
            output[output == 0] = 1

        return output

    def compute_bit_flip_rate(
        self,
        hd_vector: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> float:
        """
        Compute the bit flip rate for binary HD vectors.

        Args:
            hd_vector: Input binary HD vector
            snr_db: SNR value to use

        Returns:
            Bit flip rate (0 to 1)
        """
        if not self.binary_vectors:
            raise ValueError("Bit flip rate only applies to binary vectors")

        received = self.forward(hd_vector, snr_db)
        flips = torch.sum(hd_vector != received).item()
        total = hd_vector.numel()

        return flips / total


def create_channel(
    channel_type: str = "awgn",
    snr_db: float = 20.0,
    **kwargs
) -> WirelessChannel:
    """
    Factory function to create a wireless channel.

    Args:
        channel_type: Type of channel ("awgn", "rayleigh", "rician", "multipath")
        snr_db: Signal-to-Noise Ratio in dB
        **kwargs: Additional channel parameters

    Returns:
        Configured WirelessChannel instance

    Examples:
        >>> channel = create_channel("rayleigh", snr_db=15)
        >>> channel = create_channel("rician", snr_db=20, k_factor=5)
    """
    return WirelessChannel(
        snr_db=snr_db,
        channel_type=ChannelType(channel_type.lower()),
        **kwargs
    )


def create_image_channel(
    channel_type: str = "awgn",
    snr_db: float = 20.0,
    **kwargs
) -> ImageWirelessChannel:
    """
    Factory function to create an image wireless channel.

    Args:
        channel_type: Type of channel
        snr_db: Signal-to-Noise Ratio in dB
        **kwargs: Additional channel parameters

    Returns:
        Configured ImageWirelessChannel instance
    """
    return ImageWirelessChannel(
        snr_db=snr_db,
        channel_type=ChannelType(channel_type.lower()),
        **kwargs
    )


def create_hd_channel(
    channel_type: str = "awgn",
    snr_db: float = 20.0,
    hd_dimension: int = 10000,
    **kwargs
) -> HDVectorChannel:
    """
    Factory function to create an HD vector wireless channel.

    Args:
        channel_type: Type of channel
        snr_db: Signal-to-Noise Ratio in dB
        hd_dimension: HD vector dimension
        **kwargs: Additional channel parameters

    Returns:
        Configured HDVectorChannel instance
    """
    return HDVectorChannel(
        snr_db=snr_db,
        channel_type=ChannelType(channel_type.lower()),
        hd_dimension=hd_dimension,
        **kwargs
    )


# Convenience functions for quick testing
def awgn_channel(snr_db: float = 20.0, **kwargs) -> WirelessChannel:
    """Create an AWGN channel."""
    return create_channel("awgn", snr_db, **kwargs)


def rayleigh_channel(snr_db: float = 20.0, **kwargs) -> WirelessChannel:
    """Create a Rayleigh fading channel."""
    return create_channel("rayleigh", snr_db, **kwargs)


def rician_channel(snr_db: float = 20.0, k_factor: float = 3.0, **kwargs) -> WirelessChannel:
    """Create a Rician fading channel."""
    return create_channel("rician", snr_db, k_factor=k_factor, **kwargs)


def multipath_channel(snr_db: float = 20.0, num_taps: int = 4, **kwargs) -> WirelessChannel:
    """Create a multipath fading channel."""
    return create_channel("multipath", snr_db, num_taps=num_taps, **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("Wireless Channel Simulator Demo")
    print("=" * 50)

    # Test AWGN channel
    print("\n1. AWGN Channel Test")
    channel = awgn_channel(snr_db=10)
    signal = torch.randn(1000)
    received = channel(signal)
    print(f"   Input power: {torch.mean(signal**2):.4f}")
    print(f"   Output power: {torch.mean(received**2):.4f}")

    # Test Rayleigh channel
    print("\n2. Rayleigh Fading Channel Test")
    channel = rayleigh_channel(snr_db=15)
    received = channel(signal)
    print(f"   Input power: {torch.mean(signal**2):.4f}")
    print(f"   Output power: {torch.mean(received**2):.4f}")

    # Test image channel
    print("\n3. Image Channel Test")
    img_channel = create_image_channel("awgn", snr_db=20)
    image = torch.rand(1, 28, 28)  # Grayscale image
    noisy_image = img_channel(image)
    print(f"   Image shape: {image.shape}")
    print(f"   Output range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")

    # Test HD vector channel
    print("\n4. HD Vector Channel Test")
    hd_channel = create_hd_channel("awgn", snr_db=10, hd_dimension=10000)
    hd_vector = torch.sign(torch.randn(10000))  # Binary HD vector
    received_hv = hd_channel(hd_vector)
    flip_rate = hd_channel.compute_bit_flip_rate(hd_vector, snr_db=10)
    print(f"   HD dimension: 10000")
    print(f"   Bit flip rate at 10dB SNR: {flip_rate:.4f}")

    # BER simulation
    print("\n5. BER Simulation (BPSK over AWGN)")
    channel = awgn_channel(snr_db=10)
    snr_vals, ber_vals = channel.ber_simulation(num_bits=100000, snr_range_db=list(range(0, 15, 2)))
    for snr, ber in zip(snr_vals, ber_vals):
        print(f"   SNR={snr:2d}dB: BER={ber:.6f}")

    print("\n" + "=" * 50)
    print("Demo complete!")
