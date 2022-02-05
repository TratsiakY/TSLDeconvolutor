#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2022 Yauhen Tratsiak. All rights reserved.
# Authors: Yauhen Tratsiak <ytratsia@utk.edu>
# License: GPLv3 (GNU General Public License Version 3)
#          https://www.gnu.org/licenses/quick-guide-gplv3.html
#
# This file is part of TSL deconvolute software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

'''
This file contains small class for reading TSL data files. If should be replaced in accordance with your needs.
'''

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class TSLDataReader():
    def __init__(self, filename, mult = 10):
        self.fname = filename
        self.mult = mult
        
    def read_data(self, mult = True):
        data = []
        
        with open(self.fname, 'r') as f:
            for lines in f.readlines():
                elements = lines[:-1].split('\t')
                data.append(np.array(elements))
            data = np.array(data)
        f.close()
        
        mod_data = []
        
        for i, rows in enumerate(data):
            mod_data.append([])
            for element in rows:
                if len(element) > 1:
                    if ('PM' in element) or ('AM' in element):
                        dt = datetime.strptime(element,'%m/%d/%Y %I:%M:%S %p')
                        mod_data[i].append(np.datetime64(dt))
                    else:
                        mod_data[i].append(float(element))
                        
        mod_data = np.array(mod_data)
        init_time = mod_data[0][-1]
        
        for i in range(len(mod_data)):
            mod_data[i][-1] = ((mod_data[i][-1] - init_time)/1e+6).astype(np.float64)
        
        if mult:
            mod_data[:, 3] = mod_data[:, 3] + np.abs(min(mod_data[:, 3]))
            maxval = max(mod_data[:, 3])
            mod_data[:, 3] = mod_data[:, 3] * self.mult/maxval
        return np.array(mod_data, dtype=np.float64)

if __name__ == "__main__":
    print('Not for individual use')
