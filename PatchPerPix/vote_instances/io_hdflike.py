# credits to Saalfeld Lab
# copied and modified from https://github.com/saalfeldlab/simpleference
# try to import zarr
try:
    import zarr
    WITH_ZARR = True
except ImportError:
    WITH_ZARR = False

# try to import h5py
try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

# try to import dvid
try:
    from libdvid import DVIDNodeService
    from libdvid import ConnectionMethod
    WITH_DVID = True
except ImportError:
    WITH_DVID = False


class IoBase(object):
    """
    Base class for I/O with h5 and n5

    Arguments:
        path (str): path to h5 or n5 file
        key (str or list[str]): key or list of keys to datasets in file
        io_module (io python module): needs to follow h5py syntax.
            either z5py or h5py
        channel_orders (list[slice]): mapping of channels to output datasets (default: None)
    """
    def __init__(self, path, keys, io_module, channel_order=None):
        assert isinstance(keys, (tuple, list, str)), type(keys)
        self.path = path
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.ff = self.open(self.path)
        assert all(kk in self.ff for kk in self.keys), "%s, %s" % (self.path, self.keys)
        self.datasets = [self.ff[kk] for kk in self.keys]

        # TODO validate
        # we just assume that everything has the same shape...
        self._shape = self.datasets[0].shape

        # validate non-trivial channel orders
        if channel_order is not None:
            assert all(isinstance(cho, slice) for cho in channel_order)
            assert len(channel_order) == len(self.datasets)
            for ds, ch in zip(self.datasets, channel_order):
                n_chan = ch.stop - ch.start
                if ds.ndim == 4:
                    assert n_chan == ds.shape[0]
                elif ds.ndim == 3:
                    assert n_chan == ds.shape[0]
                    #assert n_chan == 1
                else:
                    raise RuntimeError("Invalid dataset dimensionality")
            self.channel_order = channel_order
        else:
            assert len(self.datasets) == 1, "Need channel order if given more than one dataset"
            self.channel_order = None

    def read(self, bounding_box, key):
        #assert len(self.datasets) == 1
        i = self.keys.index(key)
        return self.datasets[i][bounding_box]

    def write(self, out, out_bb):
        if self.channel_order is None:
            ds = self.datasets[0]
            assert out.ndim == ds.ndim, "%i, %i" % (out.ndim, ds.ndim)
            if out.ndim == 4:
                #ds[(slice(None),) + out_bb] = out
                ds[out_bb] = out
            else:
                ds[out_bb] = out
        else:
            for ds, ch in zip(self.datasets, self.channel_order):
                if ds.ndim == 3:
                    ds[out_bb] = out[ch][0]
                else:
                    ds[(slice(None),) + out_bb] = out[ch]

    @property
    def shape(self):
        return self._shape

    def open(self, path):
        pass

    def close(self):
        pass


class IoHDF5(IoBase):
    def __init__(self, path, keys, channel_order=None):
        assert WITH_H5PY, "Need h5py"
        super(IoHDF5, self).__init__(path, keys, h5py, channel_order)

    def open(self, path):
        return h5py.File(path, 'r')

    def close(self):
        self.ff.close()


class IoZarr(IoBase):
    def __init__(self, path, keys, channel_order=None):
        assert WITH_ZARR, "Need h5py"
        super(IoZarr, self).__init__(path, keys, zarr, channel_order)

    def open(self, path):
        return zarr.open(path)

    def close(self):
        self.ff.close()


class IoN5(IoBase):
    def __init__(self, path, keys, channel_order=None):
        assert WITH_Z5PY, "Need z5py"
        super(IoN5, self).__init__(path, keys, z5py, channel_order)


class IoDVID(object):
    def __init__(self, server_address, uuid, key):
        assert WITH_DVID, "Need dvid"
        self.ds = DVIDNodeService(server_address, uuid)
        self.key = key

        # get the shape the dvid way...
        endpoint = "/" + self.key + "/info"
        attributes = self.ds.custom_request(endpoint, "", ConnectionMethod.GET)
        # TODO do we need to increase by 1 here ?
        self._shape = tuple(mp + 1 for mp in attributes["MaxPoint"])

    def read(self, bb):
        offset = tuple(b.start for b in bb)
        shape = tuple(b.stop - b.start for b in bb)
        return self.ds.get_gray3D(self.key, shape, offset)

    def write(self, out, out_bb):
        raise NotImplementedError("Writing to DVID is not yet implemented!")

    @property
    def shape(self):
        return self._shape

    def close(self):
        pass
