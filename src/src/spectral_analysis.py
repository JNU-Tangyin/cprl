import numpy as np
from typing import Tuple, Optional
from scipy.fft import fft, fftfreq
from dataclasses import dataclass

@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results"""
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    normalized_spectrum: np.ndarray

class SpectralAnalyzer:
    def __init__(self, window_size: int = 64, step: int = 1, sampling_rate: float = 1.0):
        """
        Initialize the spectral analyzer with Wasserstein distance capabilities.
        
        Args:
            window_size: Size of the sliding window for FFT
            step: Step size for the sliding window
            sampling_rate: Sampling rate of the time series (samples per second)
        """
        self.window_size = window_size
        self.step = step
        self.sampling_rate = sampling_rate
        
    def compute_spectrum(self, signal: np.ndarray) -> SpectralAnalysisResult:
        """
        Compute the power spectrum of a signal using FFT.
        
        Args:
            signal: 1D numpy array containing the time series data
            
        Returns:
            SpectralAnalysisResult containing frequencies, power spectrum, and normalized spectrum
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        # Apply FFT
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), d=1.0/self.sampling_rate)
        
        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Normalize
        normalized_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)  # Add small epsilon to avoid division by zero
        
        return SpectralAnalysisResult(
            frequencies=freqs[:len(freqs)//2],  # Return only positive frequencies
            power_spectrum=power_spectrum[:len(freqs)//2],
            normalized_spectrum=normalized_spectrum[:len(freqs)//2]
        )
    
    def sliding_window_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute sliding window spectrum of a time series.
        
        Args:
            signal: 1D numpy array containing the time series data
            
        Returns:
            2D numpy array where each row is the normalized spectrum of a window
        """
        n_windows = max(1, (len(signal) - self.window_size) // self.step + 1)
        spectra = []
        
        for i in range(n_windows):
            window = signal[i*self.step : i*self.step + self.window_size]
            spectrum = self.compute_spectrum(window)
            spectra.append(spectrum.normalized_spectrum)
            
        return np.array(spectra)
    
    @staticmethod
    def wasserstein_distance(spectrum1: np.ndarray, spectrum2: np.ndarray, 
                           positions: Optional[np.ndarray] = None) -> float:
        """
        Compute 1D Wasserstein distance between two normalized spectra.
        
        Args:
            spectrum1: First normalized power spectrum
            spectrum2: Second normalized power spectrum
            positions: Optional array of positions/frequencies. If None, uses array indices.
            
        Returns:
            float: 1-Wasserstein distance between the two spectra
        """
        # Ensure inputs are numpy arrays
        p = np.asarray(spectrum1, dtype=np.float64)
        q = np.asarray(spectrum2, dtype=np.float64)
        
        # Normalize to probability distributions
        p = p / (np.sum(p) + 1e-10)
        q = q / (np.sum(q) + 1e-10)
        
        # If positions not provided, use array indices
        if positions is None:
            positions = np.arange(len(p))
            
        # Ensure positions is 1D and has same length as spectra
        positions = np.asarray(positions).flatten()
        if len(positions) != len(p):
            positions = np.linspace(0, 1, len(p))
            
        # Compute 1D Wasserstein distance using the closed-form solution
        # (for 1D, it's the integral of |CDF1(x) - CDF2(x)|)
        cdf1 = np.cumsum(p)
        cdf2 = np.cumsum(q)
        
        # Compute the integral using the trapezoidal rule
        return np.trapz(np.abs(cdf1 - cdf2), positions)
    
    def spectral_wasserstein_distance(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Compute Wasserstein distance between the spectral distributions of two signals.
        
        Args:
            signal1: First time series
            signal2: Second time series (should have same length as signal1)
            
        Returns:
            float: Wasserstein distance between the spectral distributions
        """
        # Compute spectra
        spec1 = self.compute_spectrum(signal1).normalized_spectrum
        spec2 = self.compute_spectrum(signal2).normalized_spectrum
        
        # Ensure both spectra have the same length
        min_len = min(len(spec1), len(spec2))
        spec1 = spec1[:min_len]
        spec2 = spec2[:min_len]
        
        # Compute Wasserstein distance
        return self.wasserstein_distance(spec1, spec2)
    
    def sliding_window_wasserstein(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """
        Compute sliding window Wasserstein distances between two time series.
        
        Args:
            signal1: First time series
            signal2: Second time series (should have same length as signal1)
            
        Returns:
            1D numpy array of Wasserstein distances for each window
        """
        # Compute sliding window spectra
        spectra1 = self.sliding_window_spectrum(signal1)
        spectra2 = self.sliding_window_spectrum(signal2)
        
        # Ensure same number of windows
        min_windows = min(len(spectra1), len(spectra2))
        spectra1 = spectra1[:min_windows]
        spectra2 = spectra2[:min_windows]
        
        # Compute Wasserstein distance for each window pair
        distances = []
        for s1, s2 in zip(spectra1, spectra2):
            dist = self.wasserstein_distance(s1, s2)
            distances.append(dist)
            
        return np.array(distances)