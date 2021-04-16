import yt
import numpy as np
import scipy.fft as fft
from yt import derived_field
import yt.units.dimensions as dimensions

class FourierData(object):
    def __init__(self, time, kx, ky, kz, FT_magnitude, FT_phase):
        # Dataset time
        self.time = time

        # Wavenumbers
        self.kx = kx
        self.ky = ky
        self.kz = kz

        # Fourier transform magnitude & phase
        self.magnitude = FT_magnitude
        self.phase = FT_phase

class EmuDataset(object):
    def __init__(self, filename):
        self.ds = yt.load(filename)
        
        self.construct_covering_grid()
        self.add_emu_fields()
        
    def construct_covering_grid(self):
        self.cg = self.ds.covering_grid(level=0, left_edge=self.ds.domain_left_edge,
                                        dims=self.ds.domain_dimensions)

        # number of cells in each dimension
        self.Nx = self.ds.domain_dimensions[0]
        self.Ny = self.ds.domain_dimensions[1]
        self.Nz = self.ds.domain_dimensions[2]

        # find dx, dy, dz in each of X,Y,Z
        # this is the spacing between cell centers in the domain
        # it is the same as the spacing between cell edges
        self.dx = (self.ds.domain_right_edge[0] - self.ds.domain_left_edge[0])/self.ds.domain_dimensions[0]
        self.dy = (self.ds.domain_right_edge[1] - self.ds.domain_left_edge[1])/self.ds.domain_dimensions[1]
        self.dz = (self.ds.domain_right_edge[2] - self.ds.domain_left_edge[2])/self.ds.domain_dimensions[2]

        if self.Nx > 1:
            # low, high edge locations in x domain
            xlo = self.ds.domain_left_edge[0]
            xhi = self.ds.domain_right_edge[0]

            # the offset between the edges xlo, xhi and the interior cell centers
            x_cc_offset = 0.5 * self.dx

            self.X, DX = np.linspace(xlo + x_cc_offset, # first cell centered location in the interior of x domain
                                xhi - x_cc_offset, # last cell centered location in the interior of x domain
                                num=self.Nx,            # Nx evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dx

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dx, DX = ", dx, DX)
            assert self.dx == DX

        if self.Ny > 1:
            # low, high edge locations in y domain
            ylo = self.ds.domain_left_edge[1]
            yhi = self.ds.domain_right_edge[1]

            # the offset between the edges ylo, yhi and the interior cell centers
            y_cc_offset = 0.5 * self.dy

            self.Y, DY = np.linspace(ylo + y_cc_offset, # first cell centered location in the interior of y domain
                                yhi - y_cc_offset, # last cell centered location in the interior of y domain
                                num=self.Ny,            # Ny evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dy

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dy, DY = ", dy, DY)
            assert self.dy == DY


        if self.Nz > 1:
            # low, high edge locations in z domain
            zlo = self.ds.domain_left_edge[2]
            zhi = self.ds.domain_right_edge[2]

            # the offset between the edges zlo, zhi and the interior cell centers
            z_cc_offset = 0.5 * self.dz

            self.Z, DZ = np.linspace(zlo + z_cc_offset, # first cell centered location in the interior of z domain
                                zhi - z_cc_offset, # last cell centered location in the interior of z domain
                                num=self.Nz,            # Nz evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dz

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dz, DZ = ", dz, DZ)
            assert self.dz == DZ
    
    def add_emu_fields(self):
        # first, define the trace
        def _make_trace(ds):
            if ('boxlib', 'N22_Re') in ds.field_list:
                def _trace(field, data):
                    return data["N00_Re"] + data["N11_Re"] + data["N22_Re"]
                return _trace
            else:
                def _trace(field, data):
                    return data["N00_Re"] + data["N11_Re"]
                return _trace

        _trace = _make_trace(self.ds)

        self.ds.add_field(("gas", "trace"), function=_trace, units="auto", dimensions=dimensions.dimensionless)

        # now, define normalized fields
        for f in self.ds.field_list:
            if "_Re" in f[1] or "_Im" in f[1]:
                fname = f[1]
                fname_norm = "{}_norm_tr".format(fname)

                def _make_derived_field(f):
                    def _derived_field(field, data):
                        return data[f]/data[("gas", "trace")]
                    return _derived_field

                _norm_derived_f = _make_derived_field(f)
                self.ds.add_field(("gas", fname_norm), function=_norm_derived_f, units="auto", dimensions=dimensions.dimensionless)

    def fourier(self, field_Re, field_Im=None, nproc=None):
        if field_Im:
            FT = np.squeeze(self.cg[field_Re][:,:,:].d + 1j * self.cg[field_Im][:,:,:].d)
        else:
            FT = np.squeeze(self.cg[field_Re][:,:,:].d)

        # use fftn to do an N-dimensional FFT on an N-dimensional numpy array
        FT = fft.fftn(FT,workers=nproc)

        # we're shifting the sampling frequencies next, so we have to shift the FFT values
        FT = fft.fftshift(FT)

        # get the absolute value of the fft
        FT_mag = np.abs(FT)

        # get the phase of the fft
        FT_phi = np.angle(FT)

        if self.Nx > 1:
            # find the sampling frequencies in X & shift them
            kx = fft.fftfreq(self.Nx, self.dx)
            kx = fft.fftshift(kx)
        else:
            kx = None

        if self.Ny > 1:
            # find the sampling frequencies in Y & shift them
            ky = fft.fftfreq(self.Ny, self.dy)
            ky = fft.fftshift(ky)
        else:
            ky = None

        if self.Nz > 1:
            # find the sampling frequencies in Z & shift them
            kz = fft.fftfreq(self.Nz, self.dz)
            kz = fft.fftshift(kz)
        else:
            kz = None

        return FourierData(self.ds.current_time, kx, ky, kz, FT_mag, FT_phi)
