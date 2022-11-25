#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

import matplotlib
import matplotlib.pyplot as plt

import string
from matplotlib import rc

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

protocol_chrono_order = ['staircaseramp1',
                         'sis',
                         'rtovmaxdiff',
                         'rvotmaxdiff',
                         'spacefill10',
                         'spacefill19',
                         'spacefill26',
                         'longap',
                         'hhbrute3gstep',
                         'hhsobol3step',
                         'wangbrute3gstep',
                         'wangsobol3step',
                         'staircaseramp2']

relabel_dict = dict(zip(protocol_chrono_order,
                        string.ascii_uppercase[:len(protocol_chrono_order)]))


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("-i", "--ignore_protocols", nargs='+', default=[])
    parser.add_argument("--model", type=str, default='Beattie')
    parser.add_argument("--figsize", "-f", nargs=2, type=int)
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--dpi", "-d", type=int, default=600)
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--share_colourbar", action='store_true')
    parser.add_argument("--file_format", default="svg")
    parser.add_argument("--separate_cbar", action='store_true')
    parser.add_argument("--remove_cbar", action='store_true')
    parser.add_argument("--fixed_params", type=int, nargs='+')

    args = parser.parse_args()

    global model_class
    model_class = common.get_model_class(args.model)

    rc('font', size=args.fontsize)
    df = pd.read_csv(args.input_file)

    df = df.sort_values('score')

    df = df[~df.fitting_protocol.isin(args.ignore_protocols)]
    df = df[~df.validation_protocol.isin(args.ignore_protocols)]

    protocols = sorted(list(df['fitting_protocol'].unique()))

    print('protocol order:', protocols)

    relabel_dict = dict(zip(protocols, string.ascii_uppercase[:len(protocols)]))

    # V for validation
    relabel_dict['longap'] = 'V'

    print(relabel_dict)

    df = df.replace({
        'validation_protocol': relabel_dict,
        'fitting_protocol': relabel_dict})

    fig = plt.figure(figsize=args.figsize)

    output_dir = common.setup_output_directory(args.output_dir, 'fitting_validation_heatmaps')
    protocol_list = df['fitting_protocol'].unique()

    if args.normalise_diagonal:
        assert False

    # if args.wells:
    #         df = df[df.well.isin(args.wells)]
    # if args.protocols:
    #     df = df[df.protocols.isin(protocols)]

    cbar_min = df['RMSE'].min()
    cbar_max = df['RMSE'].max()

    parameter_labels = model_class().get_parameter_labels()
    df[parameter_labels] = df[parameter_labels].astype(np.float64)

    if args.fixed_params:
        fixed_params = args.fixed_params
    else:
        fixed_params = list(df.fixed_param.unique())

    # Iterate over fixed parameters
    for fixed_param in fixed_params:
        lims = [df.score.min(), df.score.max()]

        if 'fixed_param' in df.columns:
            sub_df = df[df.fixed_param == fixed_param]
        else:
            sub_df = df

        fixed_param_label = model_class().get_parameter_labels()[fixed_param]
        sub_df = sub_df.sort_values(fixed_param_label)

        for j, val in enumerate(sorted(df[fixed_param_label].unique())):
            val = np.float64(val)

            if fixed_param_label in sub_df.columns:
                sub_var_df = sub_df[sub_df[fixed_param_label] == val]
            else:
                continue

            # Iterate over wells for heatmap
            for well in df['well'].unique():
                ax = fig.subplots()

                sub_well_df = sub_var_df[sub_var_df.well == well].copy()

                if args.normalise_diagonal:
                    index_df = sub_well_df.set_index(['fitting_protocol', 'validation_protocol'])
                    diagonals = {protocol: index_df.loc[(protocol, protocol), 'RMSE'] for protocol in protocol_list}
                    sub_well_df['normalised RMSE'] = [
                        row['RMSE'] /
                        diagonals[row['validation_protocol']] for _, row in sub_well_df.iterrows()]
                    # sub_well_df['log$_{10}$ normalised RMSE'] = np.log10(sub_well_df['normalised RMSE'])

                    value_col = 'normalised RMSE'

                else:
                    value_col = 'RMSE'

                pivot_df = sub_well_df.pivot(index='fitting_protocol', columns='validation_protocol',
                                             values=value_col)

                for protocol in sub_well_df.fitting_protocol.unique():
                    pivot_df.loc[protocol][protocol] = pivot_df.loc[protocol].values.min()

                pivot_df = pivot_df[np.isfinite(pivot_df)]

                if not args.vmax and not args.vmin:
                    args.vmin, args.vmax = lims

                cmap = sns.cm.mako_r
                hm = sns.heatmap(pivot_df, ax=ax, cbar_kws={'label': value_col},
                                 vmin=args.vmin, vmax=args.vmax, cmap=cmap, square=True,
                                 cbar=not args.remove_cbar, norm=matplotlib.colors.LogNorm())

                hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
                hm.set_facecolor('black')

                # if not args.alphabet_labels:
                #     hm.set_xticks([])

                # ax.set_title(f"{well}")
                ax.set_ylabel("Fitting protocol")
                ax.set_xlabel("Validation protocol")

                # fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{well}_fit_predict_heatmap_fix_{fixed_param_label}_{j}.{args.file_format}"), dpi=args.dpi)
                fig.clf()

            # Average over all wells
            lims = [df.score.min(), df.score.max()]
            ax = fig.subplots()

            sub_var_df = sub_var_df.groupby([fixed_param_label, 'fitting_protocol', 'validation_protocol']).mean().reset_index()
            sub_var_df = sub_var_df[np.isfinite(sub_var_df.score.astype(np.float64))]

            lims = sub_var_df

            if args.normalise_diagonal:
                diagonals = {protocol: index_df.loc[(protocol, protocol), 'RMSE'] for protocol in protocol_list}
                sub_df['normalised RMSE'] = [
                    row['RMSE'] /
                    diagonals[row['validation_protocol']] for _, row in sub_df.iterrows()]
                sub_df['log$_{10}$ normalised RMSE'] = np.log10(sub_df['normalised RMSE'])

                value_col = 'normalised RMSE'

            else:
                value_col = 'RMSE'

            sub_df['log RMSE'] = np.log10(sub_var_df['RMSE'])

            print(sub_var_df)
            pivot_df = sub_var_df.pivot(index='fitting_protocol', columns='validation_protocol',
                                        values=value_col)

            for protocol in sub_var_df.fitting_protocol.unique():
                pivot_df.loc[protocol][protocol] = pivot_df.loc[protocol].values.min()

            pivot_df = pivot_df[np.isfinite(pivot_df)]

            pivot_df.sort_index(level=0, ascending=True, inplace=True)
            print(pivot_df)

            if not args.vmax and not args.vmin:
                args.vmin, args.vmax = lims

            cmap = sns.cm.mako_r
            hm = sns.heatmap(pivot_df, ax=ax, cbar_kws={'label': value_col},
                             cmap=cmap,
                             square=True, cbar=not args.remove_cbar,
                             norm=matplotlib.colors.LogNorm(vmin=args.vmin, vmax=args.vmax))

            hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
            hm.set_facecolor('black')

            # if not args.alphabet_labels:
            #     hm.set_xticks([])

            ax.set_title(f"{fixed_param_label}={val:.2e}")
            ax.set_ylabel("Fitting protocol")
            ax.set_xlabel("Validation protocol")

            # fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"averaged_fit_predict_heatmap_fix_{fixed_param_label}_{j}.{args.file_format}"), dpi=args.dpi)
            fig.clf()




if __name__ == "__main__":
    main()
