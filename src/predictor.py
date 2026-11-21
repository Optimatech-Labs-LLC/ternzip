"""A shared integer probability model for encoding and decoding published matrix records."""
import struct
import numpy as np
from block_codec import Overrides
from load_kernels import corrected_intervals, interval_lookup, row_intervals
from residual_axes import decode_options, odds_table
from row_options import unpack_options

CONFIG = struct.Struct("<BBI")


class Predictor:
    def __init__(self, codec, options, rows, width, previous, bounds):
        self.previous = previous
        self.bounds = bounds
        self.mapping = self.flips = self.axes = self.row_lookup = None
        self.stages = []
        self.active = bool(codec)
        layers = []
        while codec == 4:
            layer = Overrides(options, rows, width)
            if (previous is None or previous.shape != (rows, width) or len(layers) >= 2
                    or layers and layers[0].distance != layer.distance):
                raise ValueError("The earlier-layer reference does not match this matrix")
            layers.append(layer)
            codec, options = layer.base_codec, layer.base_options
        if codec == 3:
            if previous is None or previous.shape != (rows, width):
                raise ValueError("Missing signed row source")
            _, options, self.row_odds, words, self.mapping, self.flips = unpack_options(options, rows)
        else:
            words = None
        if options:
            if codec == 0:
                raise ValueError("Unexpected matrix options")
            row_fix, col_fix, factors = decode_options(options, rows, width)
            if codec == 2:
                words = factors
            if options[0]:
                self.axes = odds_table(bounds), row_fix, col_fix
        self.word_factors = None if layers and layers[0].scalar else words
        if self.mapping is not None and self.axes is None:
            self.row_lookup = interval_lookup(bounds, self.row_odds)
        ids, flips = self.mapping, self.flips
        for layer in reversed(layers):
            if self.axes is None:
                layer.lookup = interval_lookup(bounds, layer.factors)
            self.stages.append((layer, ids, flips))
            if layer is not layers[0]:
                ids, flips = layer.next_rows(rows, width, ids, flips)

    def field(self, part, keys):
        if self.axes is None:
            site = self.bounds[keys]
        else:
            lut, row_fix, col_fix = self.axes
            site = corrected_intervals(keys, self.bounds, lut, row_fix[part], col_fix)
        field = site
        if self.mapping is not None:
            reference = self.previous[self.mapping[part]]
            signed = np.where(self.flips[part, None], 2 - reference, reference)
            field = (self.row_lookup[signed, keys] if self.row_lookup is not None
                     else row_intervals(signed, site, self.row_odds))
        for layer, ids, flips in self.stages:
            field = layer.apply(self.previous, part, site, field, ids, flips, keys)
        return field
