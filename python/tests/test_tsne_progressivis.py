from __future__ import annotations

import os
from . import ProgressiveTest, skipIf
from progressivis.core import aio
from progressivis import RandomPTable
from progressivis import CSVLoader, Constant, PTable, Sink
from progressivis.datasets import get_dataset
from progressivis.core.utils import RandomBytesIO
from progressivis.stats.tsne import TSNE
from pynene import ProgressiVisTSNE
import numpy as np

class TestTSNE(ProgressiveTest):
    def te_st_1(self) -> None:
        s = self.scheduler()
        module = CSVLoader(
            "https://aviz.fr/progressivis/mnist_784.csv.bz2",
            as_array=lambda cols: {"array": [c for c in cols if c != "class"]},
            scheduler=s,
        )
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        self.assertTrue(module.result is None)
        aio.run(s.start())
        assert module.result is not None
        table = module.result
        #import pdb;pdb.set_trace()
        tsne = ProgressiVisTSNE(table, "array", output_dims=2)
        for i in range(1000):
            tsne.run_ids(table.index)
            #print(tsne.get_y())
            print("ERROR:", i, tsne.get_error())
        self.assertEqual(len(table), 70000)
        self.assertEqual(table.columns, ["array", "class"])
        self.assertEqual(table["array"].shape, (70000, 784))
        self.assertEqual(table["class"].shape, (70000,))

    def test_2(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            "https://aviz.fr/progressivis/mnist_784.csv.bz2",
            #"/home/poli/JDF2016/github/PANENE/python/tests/mnist_784.csv",
            as_array=lambda cols: {"array": [c for c in cols if c != "class"]},
            scheduler=s, sep=" "
        )
        random = RandomPTable(1, rows=1000, throttle=1, scheduler=s)
        tsne = TSNE(array_col="array", output_cols=["x", "y"], scheduler=s)
        tsne.input.table = csv.output.result
        tsne.input.pulse = random.output.result
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = tsne.output.result
        self.assertTrue(tsne.result is None)
        aio.run(s.start())
        assert tsne.result is not None
        table = tsne.result
        #import pdb;pdb.set_trace()
        #self.assertEqual(len(table), 70000)
        self.assertEqual(table.columns, ["x", "y"])
        arr = tsne.tsne.get_y()
        np.savetxt("/tmp/foo.csv", arr, delimiter=" ")
        #tsne.tsne.dump_y()
