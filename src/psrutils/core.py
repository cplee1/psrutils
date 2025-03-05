from typing import Union

import psrchive

from psrutils import C0

__all__ = ["StokesCube"]


class StokesCube(object):
    """Wrapper for a PSRCHIVE archive, stored in the Stokes basis."""

    def __init__(
        self,
        archive: psrchive.Archive,
        clone: bool = False,
        tscrunch: int | None = None,
        fscrunch: int | None = None,
        bscrunch: int | None = None,
        rotate_phase: float | None = None,
    ):
        """Create a StokesCube instance from a PSRCHIVE archive.

        Parameters
        ----------
        archive : `psrchive.Archive`
            An Archive object.
        clone : `bool`, optional
            If True, clone the input object. Otherwise store a reference to 'archive'.
            Default: `False`.
        tscrunch : `int`, optional
            Scrunch in time to this number of sub-integrations. Default: `None`.
        fscrunch : `int`, optional
            Scrunch in frequency to this number of channels. Default: `None`.
        bscrunch : `int`, optional
            Scrunch in phase to this number of bins. Default: `None`.
        rotate_phase : `float`, optional
            Rotate in phase by this amount. Default: `None`.
        """
        if type(archive) is not psrchive.Archive:
            raise ValueError("archive must be a psrchive.Archive")

        if clone:
            self._archive = archive.clone()
        else:
            self._archive = archive

        # Ensure the archive is in the Stokes basis
        if self._archive.get_state() != "Stokes" and self._archive.get_npol() == 4:
            try:
                self._archive.convert_state("Stokes")
            except RuntimeError:
                print("Could not convert to Stokes.")

        # Ensure the archive is dedispersed
        if not self._archive.get_dedispersed():
            try:
                self._archive.dedisperse()
            except RuntimeError:
                print("Could not dedisperse archive.")

        # Must remove the baseline before downsampling
        try:
            self._archive.remove_baseline()
        except RuntimeError:
            print("Could not remove baseline from archive.")

        # Downsample
        if type(tscrunch) is int:
            if tscrunch < self._archive.get_nsubint():
                self._archive.tscrunch_to_nsub(tscrunch)
        if type(fscrunch) is int:
            if fscrunch < self._archive.get_nchan():
                self._archive.fscrunch_to_nchan(fscrunch)
        if type(bscrunch) is int:
            if bscrunch < self._archive.get_nbin():
                self._archive.bscrunch_to_nbin(bscrunch)

        # Rotate
        if type(rotate_phase) is float:
            self._archive.rotate_phase(rotate_phase)

    @property
    def ctr_freq(self):
        """Centre frequency in MHz."""
        return self._archive.get_centre_frequency()

    @property
    def min_freq(self):
        """Lower edge frequency in MHz."""
        return self._archive.get_centre_frequency() - abs(self._archive.get_bandwidth()) / 2.0

    @property
    def max_freq(self):
        """Upper edge frequency in MHz."""
        return self._archive.get_centre_frequency() + abs(self._archive.get_bandwidth()) / 2.0

    @property
    def freqs(self):
        """The centre frequencies of all subbands in MHz."""
        return self._archive.get_frequencies() * 1e6

    @property
    def lambda_sq(self):
        """The squared centre wavelengths of all subbands in m^2."""
        freqs = self._archive.get_frequencies() * 1e6
        return (C0 / freqs) ** 2

    @property
    def int_time(self):
        """Total integration time in seconds."""
        start_time = self._archive.get_first_Integration().get_start_time()
        end_time = self._archive.get_last_Integration().get_end_time()
        int_time = end_time - start_time
        return int_time.in_seconds()

    @property
    def num_subint(self):
        """The number of subintegration in the archive."""
        return self._archive.get_nsubint()

    @property
    def num_subband(self):
        """The number of frequency subbands in the archive."""
        return self._archive.get_nchan()

    @property
    def num_bin(self):
        """The number of phase bins in the archive."""
        return self._archive.get_nbin()

    @property
    def archive(self):
        """A reference to the stored PSRCHIVE archive object."""
        return self._archive

    @property
    def archive_clone(self):
        """A clone of the stored PSRCHIVE archive object."""
        return self._archive.clone()

    @property
    def subints(self):
        """Average in frequency. Output has dimensions (time, pol, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        return tmp_archive.get_data()[:, :, 0, :]

    @property
    def subbands(self):
        """Average in time. Output has dimensions (pol, freq, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        return tmp_archive.get_data()[0, :, :, :]

    @property
    def mean_subband(self):
        """Average in time and phase. Output has dimensions (pol, freq)."""
        tmp_archive = self._archive.clone()
        tmp_archive.tscrunch()
        tmp_archive.bscrunch_to_nbin(1)
        return tmp_archive.get_data()[0, :, :, 0]

    @property
    def pol_profile(self):
        """Average in frequency and time. Output has dimensions (pol, phase)."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        return tmp_archive.get_data()[0, :, 0, :]

    @property
    def profile(self):
        """Average in frequency, time, and polarisation. Output is a 1D array."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        tmp_archive.pscrunch()
        return tmp_archive.get_data()[0, 0, 0, :]

    @property
    def snr(self):
        """Get the S/N of the integrated profile."""
        tmp_archive = self._archive.clone()
        tmp_archive.fscrunch()
        tmp_archive.tscrunch()
        tmp_archive.pscrunch()
        prof = tmp_archive.get_Profile(0, 0, 0)
        return prof.snr()

    @property
    def source(self):
        """Get the source name."""
        return self._archive.get_source()

    def bscrunch_to_nbin(self, nbin: int):
        """ "Downsample to nbin phase bins."""
        self._archive.bscrunch_to_nbin(nbin)

    def rotate_phase(self, phase: float):
        """Rotate by a fraction of the pulse phase."""
        self._archive.rotate_phase(phase)

    def defaraday(self, rm: float):
        """De-Faraday rotate to an RM."""
        self._archive.set_rotation_measure(rm)

    @classmethod
    def from_psrchive(
        cls,
        archive: Union[str, psrchive.Archive],
        clone: bool = False,
        tscrunch: int | None = None,
        fscrunch: int | None = None,
        bscrunch: int | None = None,
        rotate_phase: float | None = None,
    ):
        """Create a StokesCube from a PSRCHIVE archive object.

        Parameters
        ----------
        archive : `str` or `psrchive.Archive`
            Path to an archive file, or an Archive object, to load.
        clone : `bool`, optional
            If True and a `psrchive.Archive` object is provided, clone the input
            object. Otherwise store a reference to 'archive'. Default: `False`.
        tscrunch : `int`, optional
            Scrunch in time to this number of sub-integrations. Default: `None`.
        fscrunch : `int`, optional
            Scrunch in frequency to this number of channels. Default: `None`.
        bscrunch : `int`, optional
            Scrunch in phase to this number of bins. Default: `None`.
        rotate_phase : `float`, optional
            Rotate in phase by this amount. Default: `None`.

        Returns
        -------
        cube : `StokesCube`
            A StokesCube object.
        """
        if type(archive) is str:
            archive = psrchive.Archive.load(archive)
            clone = False

        return cls(archive, clone, tscrunch, fscrunch, bscrunch, rotate_phase)
