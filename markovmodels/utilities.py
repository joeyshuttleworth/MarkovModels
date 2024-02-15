#!/usr/bin/env python3

import datetime
import os
import subprocess
import sys
import uuid
import json

import numpy as np
import pandas as pd
import regex as re

from pcpostprocess.voltage_protocols import VoltageProtocol


def calculate_reversal_potential(T=293, K_in=120, K_out=5):
    """
    Compute the Nernst potential of a potassium channel.

    """
    # E is the Nernst potential for potassium ions across the membrane
    # Gas constant R, temperature T, Faradays constat F
    R = 8.31455
    F = 96485

    # valency of ions (1 in the case of K^+)
    z = 1

    # Nernst potential
    E = R * T / (z * F) * np.log(K_out / K_in)

    # Convert to mV
    return E * 1e3


def get_data(well, protocol, data_directory, experiment_name='',
             label='', sweep=None):

    if not label:
        label = ''

    # Find data
    if sweep:
        if label:
            label = label + '-'
        else:
            label = ''

    regex = re.compile(f"^{experiment_name}-{protocol}-{well}-{label}sweep{sweep}-subtracted.csv$")

    print(regex, data_directory)

    fname = next(filter(regex.match, os.listdir(data_directory)))
    data = pd.read_csv(os.path.join(data_directory, fname),
                       float_precision='round_trip').values

    if len(data.T) > 1:
        raise ValueError(f"shape of values is {data.shape} when it should be 1d")

    v_regex = re.compile(f"^{experiment_name}-{protocol}.json$")
    protocols_dir = os.path.join(data_directory, 'protocols')
    print(v_regex, protocols_dir)

    fname = next(filter(v_regex.match, os.listdir(protocols_dir)))

    with open(os.path.join(protocols_dir, fname), 'r') as fin:
        json_contents = json.load(fin)

    voltage_protocol = VoltageProtocol.from_json(json_contents)

    return data.flatten(), voltage_protocol


def get_all_wells_in_directory(data_dir, experiment_name='newtonrun4'):

    regex = f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z]|0-9]*)"
    regex = re.compile(regex)
    wells = []
    group = 1

    for f in filter(regex.match, os.listdir(data_dir)):
        well = re.search(regex, f).groups()[group]
        wells.append(well)

    wells = list(set(wells))
    return wells


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def setup_output_directory(dirname: str = None, subdir_name: str = None):

    if dirname is None:
        if subdir_name:
            dirname = os.path.join("output", f"{subdir_name}-{uuid.uuid4()}")
        else:
            dirname = os.path.join("output", f"output-{uuid.uuid4()}")

    if subdir_name is not None:
        dirname = os.path.join(dirname, subdir_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'info.txt'), 'w') as description_fout:
        git_hash = get_git_revision_hash()
        datetimestr = str(datetime.datetime.now())
        description_fout.write(f"Date: {datetimestr}\n")
        description_fout.write(f"Commit {git_hash}\n")
        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")

    return dirname


def put_copy(arr, ind, v, mode='raise'):
    _arr = arr.copy()
    np.put(_arr, ind, v, mode)
    return _arr
