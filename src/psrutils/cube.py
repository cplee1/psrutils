import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from psrchive import Archive
from scipy.constants import speed_of_light

__all__ = ["StokesCube"]

logger = logging.getLogger(__name__)


class StokesCube(object):
    """Wrapper for a PSRCHIVE Archive stored in the Stokes basis.

    Parameters
    ----------
    archive : Archive
        An Archive object.
    clone : bool, default: False
        If True, will clone the input Archive.
    tscrunch : int or None, default: None
        Average in time to this number of sub-integrations.
    fscrunch : int or None, default: None
        Average in frequency to this number of channels.
    bscrunch : int or None, default: None
        Average in phase to this number of bins.
    rotate_phase : float or None, default: None
        Rotate in phase by this many rotations.

    Attributes
    ----------
    ctr_freq
    min_freq
    max_freq
    freqs
    lambda_sq
    int_time
    start_mjd
    end_mjd
    num_subint
    num_subband
    num_bin
    num_pol
    archive
    archive_clone
    subints
    subbands
    mean_subbands
    pol_profile
    profile
    snr
    source
    """

    def __init__(
        self,
        archive: Archive,
        clone: bool = False,
        tscrunch: int | None = None,
        fscrunch: int | None = None,
        bscrunch: int | None = None,
        rotate_phase: float | None = None,
    ) -> None:
        """Create a StokesCube from a PSRCHIVE Archive.

        Raises
        ------
        ValueError
            If the archive is not an instance of the Archive class.
        """
        if not isinstance(archive, Archive):
            raise ValueError("'archive' must be an instance of psrchive.Archive.")

        if clone:
            self._archive = archive.clone()
        else:
            self._archive = archive

        # Ensure the archive is in the Stokes basis
        if self._archive.get_state() != "Stokes" and self._archive.get_npol() == 4:
            self._archive.convert_state("Stokes")

        # Ensure the archive is dedispersed
        if not self._archive.get_dedispersed():
            self._archive.dedisperse()

        # Must remove the baseline before downsampling
        self._archive.remove_baseline()

        # Downsample
        if isinstance(tscrunch, int):
            if tscrunch < self._archive.get_nsubint():
                self._archive.tscrunch_to_nsub(tscrunch)
        if isinstance(fscrunch, int):
            if fscrunch < self._archive.get_nchan():
                self._archive.fscrunch_to_nchan(fscrunch)
        if isinstance(bscrunch, int):
            if bscrunch < self._archive.get_nbin():
                self._archive.bscrunch_to_nbin(bscrunch)

        # Rotate
        if isinstance(rotate_phase, float):
            self._archive.rotate_phase(rotate_phase)

    @property
    def ctr_freq(self) -> float:
        """float: Centre frequency in MHz."""
        return self._archive.get_centre_frequency()

    @property
    def min_freq(self) -> float:
        """float: Lower edge frequency in MHz."""
        return (
            self._archive.get_centre_frequency()
            - abs(self._archive.get_bandwidth()) / 2.0
        )

    @property
    def max_freq(self) -> float:
        """float: Upper edge frequency in MHz."""
        return (
            self._archive.get_centre_frequency()
            + abs(self._archive.get_bandwidth()) / 2.0
        )

    @property
    def freqs(self) -> NDArray[np.float_]:
        """ndarray[float]: Subband centre frequencies in MHz."""
        return self._archive.get_frequencies() * 1e6

    @property
    def lambda_sq(self) -> NDArray[np.float_]:
        """ndarray[float]: Squared subband centre wavelengths in m^2."""
        freqs = self._archive.get_frequencies() * 1e6
        return (speed_of_light / freqs) ** 2

    @property
    def int_time(self) -> float:
        """float: Total integration time in seconds."""
        start_time = self._archive.get_first_Integration().get_start_time()
        end_time = self._archive.get_last_Integration().get_end_time()
        int_time = end_time - start_time
        return int_time.in_seconds()

    @property
    def start_mjd(self) -> float:
        """float: Start time in MJD."""
        return self._archive.get_first_Integration().get_start_time().in_days()

    @property
    def end_mjd(self) -> float:
        """float: End time in MJD."""
        return self._archive.get_last_Integration().get_end_time().in_days()

    @property
    def num_subint(self) -> int:
        """int: Number of subintegrations."""
        return self._archive.get_nsubint()

    @property
    def num_subband(self) -> int:
        """int: Number of frequency subbands."""
        return self._archive.get_nchan()

    @property
    def num_bin(self) -> int:
        """int: Number of phase bins."""
        return self._archive.get_nbin()

    @property
    def num_pol(self) -> int:
        """int: Number of instrumental polarisations."""
        return self._archive.get_npol()

    @property
    def archive(self) -> Archive:
        """Archive: The stored Archive object."""
        return self._archive

    @property
    def archive_clone(self) -> Archive:
        """Archive: A clone of the stored Archive object."""
        return self._archive.clone()

    @property
    def subints(self) -> NDArray[np.float32]:
        """ndarray[float32]: The subintegrations obtained by averaging the
        data in frequency. Output has dimensions (time, pol, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        return tmp_archive.get_data()[:, :, 0, :]

    @property
    def subbands(self) -> NDArray[np.float32]:
        """ndarray[float32]: The subbands obtained by averaging the data in
        time. Output has dimensions (pol, freq, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        return tmp_archive.get_data()[0, :, :, :]

    @property
    def mean_subbands(self) -> NDArray[np.float32]:
        """ndarray[float32]: The subbands obtained by averaging in time and
        phase. Output has dimensions (pol, freq)."""
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        tmp_archive.bscrunch_to_nbin(1)
        return tmp_archive.get_data()[0, :, :, 0]

    @property
    def pol_profile(self) -> NDArray[np.float32]:
        """ndarray[float32]: The full-Stokes profile obtained by averaging
        in frequency and time. Output has dimensions (pol, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        return tmp_archive.get_data()[0, :, 0, :]

    @property
    def profile(self) -> NDArray[np.float32]:
        """ndarray[float32]: The profile obtained by averaging in
        frequency, time, and polarisation. Output is a 1D array."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        tmp_archive.pscrunch()
        return tmp_archive.get_data()[0, 0, 0, :]

    @property
    def snr(self) -> float:
        """float: The S/N of the integrated Stokes I profile calculated by
        PSRCHIVE."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        tmp_archive.pscrunch()
        prof = tmp_archive.get_Profile(0, 0, 0)
        return prof.snr()

    @property
    def source(self) -> str:
        """str: The source name."""
        return self._archive.get_source()

    def bscrunch_to_nbin(self, nbin: int) -> None:
        """Downsample along the phase axis.

        Parameters
        ----------
        nbin : int
            The number of phase bins to downsample to.
        """
        self._archive.bscrunch_to_nbin(nbin)

    def rotate_phase(self, phase: float) -> None:
        """Rotate along the phase axis.

        Parameters
        ----------
        phase : float
            The number of rotations to rotate by.
        """
        self._archive.rotate_phase(phase)

    def defaraday(self, rm: float) -> None:
        """Incoherently de-rotate the data to a given RM.

        rm : float
            The rotation measure in rad/m^2.
        """
        self._archive.set_rotation_measure(rm)

    @classmethod
    def from_psrchive(
        cls,
        archive: str | Archive,
        clone: bool = False,
        tscrunch: int | None = None,
        fscrunch: int | None = None,
        bscrunch: int | None = None,
        rotate_phase: float | None = None,
    ):
        """Create a StokesCube from a PSRCHIVE archive object.

        Parameters
        ----------
        archive : str or Archive
            Path to an archive file or an Archive object to load.
        clone : bool, default: False
            If True, will clone the input Archive.
        tscrunch : int or None, default: None
            Average in time to this number of sub-integrations.
        fscrunch : int or None, default: None
            Average in frequency to this number of channels.
        bscrunch : int or None, default: None
            Average in phase to this number of bins.
        rotate_phase : float or None, default: None
            Rotate in phase by this many rotations.

        Returns
        -------
        StokesCube
            A StokesCube object containing the provided Archive.
        """
        if type(archive) is str:
            archive = Archive.load(archive)
            clone = False

        archive = cast(Archive, archive)

        return cls(archive, clone, tscrunch, fscrunch, bscrunch, rotate_phase)
