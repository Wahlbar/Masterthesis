"""Training of all models for the paper"""
""" I added mostly cosmetic changes to have a good position of the legend, when plotting larger sets of loss functions.
These are all manual changes. An automation could be done for better improvements."""

import argparse
import multiprocessing
import collections
import subprocess
import pathlib
import openset_imagenet
import os
import torch
import numpy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pathlib import Path
from matplotlib import pyplot, cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (1,2,3),
        help="Select the protocols that should be evaluated"
    )
    parser.add_argument(
        "--loss-functions", "-l",
        nargs = "+",
        choices = ('softmax', 'garbage', 'BG B', 'BG 1', 'BG F', 'BG FK', 'BG FN', 'entropic', 'EOS 1', 'EOS BN', 'EOS 0.5', 'EOS 0.1', 'EOS BC', 'EOS FM', 'EOS FS', 'EOS FK', 'EOS FN'),
        default = ('softmax', 'garbage', 'BG B', 'BG 1', 'BG F', 'BG FK', 'BG FN', 'entropic', 'EOS 1', 'EOS BN', 'EOS 0.5', 'EOS 0.1', 'EOS BC', 'EOS FM', 'EOS FS', 'EOS FK', 'EOS FN'),
        help = "Select the loss functions that should be evaluated"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices = ("S", "BG", "BG B", "BG 1", "BG F", "BG FK", "BG FN", "EOS", "EOS 1", "EOS BN", "EOS 0.5", "EOS 0.1", "EOS BC", 'EOS FM', 'EOS FS', 'EOS FK', 'EOS FN'),
        default = ("S", "BG", "BG B", "BG 1", "BG F", "BG FK", "BG FN", "EOS", "EOS 1", "EOS BN", "EOS 0.5", "EOS 0.1", "EOS BC", 'EOS FM', 'EOS FS', 'EOS FK', 'EOS FN'),
        help = "Select the labels for the plots"
    )
    parser.add_argument(
        "--use-best",
        action = "store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )
    # TODO: What should be the usage of this?
    parser.add_argument(
        "--force", "-f",
        action = "store_true",
        help = "If set, score files will be recomputed even if they already exist"
    )

    # Probably not needed
    parser.add_argument(
      "--linear",
      action="store_true",
      help = "If set, OSCR curves will be plot with linear FPR axis"
    )

    # Eventually if I run multiple Protocols
    parser.add_argument(
      "--sort-by-loss", "-s",
      action = "store_true",
      help = "If selected, the plots will compare across protocols and not across algorithms"
    )
    parser.add_argument(
        "--output-directory", "-o",
        type=pathlib.Path,
        default="experiments",
        help="Directory where the models are saved"
    )
    parser.add_argument(
        "--imagenet-directory",
        type=pathlib.Path,
        default=pathlib.Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=pathlib.Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )
    parser.add_argument(
      "--plots-directory",
      default = "evaluation/",
      help = "Select where to write the plots into"
    )
    parser.add_argument(
      "--table-directory",
      default = "evaluation/",
      help = "Select where to write the tables into"
    )

    # Added folder pathes to save everything in the results folder
    args = parser.parse_args()

    args.output_directory = Path(args.output_directory)
    args.plots_directory = Path(args.plots_directory)
    args.table_directory = Path(args.table_directory)

    suffix = 'linear' if args.linear else 'best' if args.use_best else 'last'
    if args.sort_by_loss:
      suffix += "_by_loss"
    args.file_name = f"Results_{suffix}_{args.labels}.pdf"
    args.file_path = args.plots_directory / args.file_name
    args.table_name = f"Results_{suffix}_{args.labels}.tex"
    args.table_path = args.table_directory / args.table_name
    return args


def load_scores(args):
    # collect all result files
    scores = {p:{} for p in args.protocols}
    epoch = {p:{} for p in args.protocols}
    for protocol in args.protocols:
      for loss in args.loss_functions:
        experiment_dir = args.output_directory / f"Protocol_{protocol}/{loss}"
        suffix = "_best" if args.use_best else "_curr"
        checkpoint_file = experiment_dir / (loss+suffix+".pth")
        score_files = {v : experiment_dir / f"{loss}_{v}_arr{suffix}.npz" for v in ("val", "test")}
        if os.path.exists(checkpoint_file):
          if not all(os.path.exists(v) for v in score_files.values()) or args.force:
            # extract score files first
            print("Extracting scores of", checkpoint_file)
            call = ["evaluate_imagenet.py", loss, str(protocol), "--output-directory", experiment_dir, "--imagenet-directory", args.imagenet_directory, "--protocol-directory", args.protocol_directory]
            if args.gpu is not None:
              call += ["-g", str(args.gpu)]
            if args.use_best:
              call += ["-b"]
            subprocess.call(call)
          # remember files
          scores[protocol][loss] = openset_imagenet.util.read_array_list(score_files)

          # remember epoch where scores are extracted
          checkpoint = torch.load(checkpoint_file, map_location="cpu")
          epoch[protocol][loss] = (checkpoint["epoch"],checkpoint["best_score"])

        else:
          print ("Checkpoint file", checkpoint_file, "not found, skipping protocol", protocol, loss)
          scores[protocol][loss] = None
          epoch[protocol][loss] = (0, 0)
    return scores, epoch

def plot_OSCR(args, scores):
    # plot OSCR
    P = len(args.protocols)
    # For horizontal alignment
    # fig = pyplot.figure(figsize=(5*P,6))
    # gs = fig.add_gridspec(2, P, hspace=0.2, wspace=0.1)
    # For vertical alignment
    fig = pyplot.figure(figsize=(12,3*P+1))
    gs = fig.add_gridspec(P, 2, hspace=0.2, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat
    font = 15
    scale = 'linear' if args.linear else 'semilog'

    if args.sort_by_loss:
      for index, l in enumerate(args.loss_functions):
        # For valudation set
        # val = [scores[p][l]["val"] if scores[p][l] is not None else None for p in args.protocols]
        test = [scores[p][l]["test"] if scores[p][l] is not None else None for p in args.protocols]
        openset_imagenet.util.plot_oscr(arrays=test, methods=[l]*len(args.protocols), scale=scale, title=f'{args.labels[index]} Negative',
                      ax_label_font=font, ax=axs[index], unk_label=-1,)
        openset_imagenet.util.plot_oscr(arrays=test, methods=[l]*len(args.protocols), scale=scale, title=f'{args.labels[index]} Unknown',
                      ax_label_font=font, ax=axs[index+P], unk_label=-2,)
      # Manual legend for single protocol plots
      if len(args.protocols) == 1:
        axs[-P].legend([f"$P_{p}$" for p in args.protocols], frameon=False,
                fontsize=font - 1, bbox_to_anchor=(0.8, -0.2), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
      else:
        axs[-P].legend([f"$P_{p}$" for p in args.protocols], frameon=False,
                fontsize=font - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
    else:
      for index, p in enumerate(args.protocols):
#        val = [scores[p][l]["val"] if scores[p][l] is not None else None for l in args.loss_functions]
        test = [scores[p][l]["test"] if scores[p][l] is not None else None for l in args.loss_functions]
        # for vertical alignment
        openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'$P_{p}$ Negative',
                      ax_label_font=font, ax=axs[2*index], unk_label=-1,)
        openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'$P_{p}$ Unknown',
                      ax_label_font=font, ax=axs[2*index+1], unk_label=-2,)
        # for horizontal alignment

        # openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'$P_{p}$ Negative',
        #               ax_label_font=font, ax=axs[index], unk_label=-1,)
        # openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'$P_{p}$ Unknown',
        #               ax_label_font=font, ax=axs[index+P], unk_label=-2,)

      # Manual legend for single protocols plots
      if len(args.protocols) == 1:
         axs[-P].legend(args.labels, frameon=False,
                fontsize=font - 1, bbox_to_anchor=(0.8, -0.2), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
      else:
        # For horizontal alignment
        # axs[-P].legend(args.labels, frameon=False,
        #         fontsize=font - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
        # For vertical alignment & 5 protocols
        axs[-P].legend(args.labels, frameon=False,
            fontsize=font - 1, bbox_to_anchor=(-0.2, -1.3), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
    # Axis properties
    for ax in axs:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

    # Figure labels
    if len(args.protocols) == 1:
      # For horizontal alignment
      # fig.text(0.5, 0.025, 'FPR', ha='center', fontsize=font)
      # fig.text(-0.01, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
      # For vertical alignment
      fig.text(0.5, 0.025, 'FPR', ha='center', fontsize=font)
      fig.text(-0.01, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
    else:
      # For horizontal alignment
      # fig.text(0.5, 0.03, 'FPR', ha='center', fontsize=font)
      # fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
      # For vertical alignment
      fig.text(0.5, 0.05, 'FPR', ha='center', fontsize=font)
      fig.text(0.065, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

def plot_confidences(args):

  # locate event paths
  event_files = {p:collections.defaultdict(list) for p in args.protocols}
  for protocol in args.protocols:
    protocol_dir = args.output_directory/f"Protocol_{protocol}"
    for loss_function in args.loss_functions:
      loss_dir = protocol_dir/f"{loss_function}"
      if os.path.exists(loss_dir):
        files = sorted(os.listdir(loss_dir))
        # get the event files
        for f in files:
          for l in args.loss_functions:
            if f.startswith("event") and f.endswith(l + ".log"):
              loss = f.split("-")[1].split(".")[0]
              # set (overwrite) event file
              event_files[protocol][loss].append(loss_dir / f)
  P = len(args.protocols)
  linewidth = 1.5
  legend_pos = "lower right"
  font_size = 15
  color_palette = cm.get_cmap('tab10', 10).colors
  # For vertical alignment
  fig = pyplot.figure(figsize=(12,3*P+1))
  gs = fig.add_gridspec(P, 2, hspace=0.2, wspace=0.1)
  # For horizontal alignment
  # fig = pyplot.figure(figsize=(5*P, 6))
  # gs = fig.add_gridspec(2, P, hspace=0.2, wspace=0.1)
  axs = gs.subplots(sharex=True, sharey=True)
  axs = axs.flat


  def load_accumulators(files):
    known_data, unknown_data = {}, {}
    for event_file in files:
      try:
        event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
        event_acc.Reload()
        for event in event_acc.Scalars('val/conf_kn'):
          known_data[event.step+1] = event.value
        for event in event_acc.Scalars('val/conf_unk'):
          unknown_data[event.step+1] = event.value
      except KeyError: pass

    # re-structure
    return zip(*sorted(known_data.items())), zip(*sorted(unknown_data.items()))

  max_len = 0
  min_len = 100
  for index, protocol in enumerate(args.protocols):
      # for vertical alignment
      ax_kn = axs[2 * index]
      ax_unk = axs[2 * index + 1]
      # for horizontal alignment
      # ax_kn = axs[index]
      # ax_unk = axs[index + P]
      for c, loss in enumerate(args.loss_functions):
        step_kn, val_kn, step_unk, val_unk = [[]]*4

        if loss in event_files[protocol]:
          # Read data from the tensorboard object
          (step_kn, val_kn), (step_unk, val_unk) = load_accumulators(event_files[protocol][loss])
        else:
          step_kn, val_kn, step_unk, val_unk = [[]]*4

        # Plot known confidence
        ax_kn.plot(step_kn, val_kn, linewidth=linewidth, label = loss + ' kn', color=color_palette[c])
        # Plot unknown confidence
        ax_unk.plot(step_unk, val_unk, linewidth=linewidth, label = loss + ' unk', color=color_palette[c])
        if len(step_kn):
          max_len = max(max_len, max(step_kn))
          min_len = min(min_len, min(step_kn))
      # set titles
      ax_kn.set_title(f"$P_{protocol}$ Known", fontsize=font_size)
      ax_unk.set_title(f"$P_{protocol}$ Negative", fontsize=font_size)

  # Manual legend for horizontal alignment
  # axs[-2].legend(args.labels, frameon=False,
  #               fontsize=font_size - 1, bbox_to_anchor=(0.8, -0.1), ncol=3, handletextpad=0.5, columnspacing=1)

  # Manual legend
  if len(args.protocols) == 2:
    axs[-P].legend(args.labels, frameon=False,
            fontsize=font_size - 1, bbox_to_anchor=(0.8, -0.2), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
  else:
    # for 3 protocols
    axs[-P].legend(args.labels, frameon=False,
            fontsize=font_size - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
    # for 5 protocols
    axs[-P].legend(args.labels, frameon=False,
            fontsize=font_size - 1, bbox_to_anchor=(-0.2, -1.3), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)


  for ax in axs:
      # set the tick parameters for the current axis handler
      ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
      ax.set_xlim(min_len, max_len)
      ax.set_ylim(0, 1)
      # Thicklocator parameters
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
      ax.xaxis.set_major_locator(MaxNLocator(6))
      ax.label_outer()
  # X label
  if len(args.protocols) == 1:
    # For vertical alignment
    fig.text(0.5, 0, 'Epoch', ha='center', fontsize=font_size)
    # For horizontal alignment
    # fig.text(0.5, 0.025, 'Epoch', ha='center', fontsize=font_size)
  else:
    # For vertical alignment
    fig.text(0.5, 0.05, 'Epoch', ha='center', fontsize=font_size)
    fig.text(0.065, 0.5, 'Confidence', va='center', rotation='vertical', fontsize=font_size)
    # For horizontal alignment
    # fig.text(0.5, 0.03, 'Epoch', ha='center', fontsize=font_size)
    # fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)


# Softmax histogramms
def plot_softmax(args, scores):

    font_size = 15
    bins = 30
    P = len(args.protocols)
    N = len(args.loss_functions)

    fig = pyplot.figure(figsize=(3*N+1, 2*P))
    gs = fig.add_gridspec(P, N, hspace=0.2, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    axs = axs.flat
    # Manual colors
    edge_unk = colors.to_rgba('indianred', 1)
    fill_unk = colors.to_rgba('firebrick', 0.04)
    edge_neg = colors.to_rgba('tab:blue', 1)
    fill_neg = colors.to_rgba('tab:blue', 0.04)
    edge_kn = colors.to_rgba('tab:green', 1)
    fill_kn = colors.to_rgba('tab:green', 0.04)

    index = 0
    for protocol in args.protocols:
      for l, loss in enumerate(args.loss_functions):
        # Calculate histogram
        drop_bg = loss in ["garbage", "BG B", "BG 1", "BG F", "BG FK", "BG FN"]  #  Drop the background class
        if scores[protocol][loss] is not None:
          kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
              scores[protocol][loss]["test"],
              unk_label=-2, # for unknown samples
              metric='score',
              bins=bins,
              drop_bg=drop_bg
          )
          kn_hist, kn_edges, neg_hist, neg_edges = openset_imagenet.util.get_histogram(
              scores[protocol][loss]["test"],
              unk_label=-1, # for negative samples
              metric='score',
              bins=bins,
              drop_bg=drop_bg
          )
        else:
          kn_hist, kn_edges, unk_hist, unk_edges, neg_hist, neg_edges = [], [0], [], [0], [], [0]

        # Plot histograms
        axs[index].stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
        axs[index].stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)
        axs[index].stairs(neg_hist, neg_edges, fill=True, color=fill_neg, edgecolor=edge_neg, linewidth=1)
        # axs[ix].set_yscale('log')
        axs[index].set_title(f"$P_{{{protocol}}}$ {args.labels[l]}")
        index += 1

    # Share y axis of the histograms of the same protocol
    for p in range(P):
      for l in range(1,N):
        axs[N*p+l-1].sharey(axs[N*p+l])

    for ax in axs:
        # set the tick parameters for the current axis handler
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.label_outer()

    # Manual legend 
    # For one protocol
    if len(args.protocols) == 1:
      axs[-2].legend(['Known', 'Unknown', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(0.8, -1.6),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # For four protocols
    elif len(args.loss_functions) == 4:
      axs[-2].legend(['Known', 'Unknown', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(-1, -0.05),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # For five protocols
    elif len(args.loss_functions) == 5:
      axs[-2].legend(['Known', 'Unknown', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(-2, -0.1),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    else:
      axs[-2].legend(['Known', 'Unknown', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(0.2, -0.08),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # X label
    fig.text(0.5, 0.02, 'Test Set', ha='center', fontsize=font_size)

# Softmax histogramms
# Added to also plot the validation set histogrm. Can be ignored.
def plot_softmax_validation(args, scores):

    font_size = 15
    bins = 30
    P = len(args.protocols)
    N = len(args.loss_functions)

    fig = pyplot.figure(figsize=(3*N+1, 2*P))
    gs = fig.add_gridspec(P, N, hspace=0.2, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    axs = axs.flat
    # Manual colors
    edge_unk = colors.to_rgba('tab:blue', 1)
    fill_unk = colors.to_rgba('tab:blue', 0.04)
    edge_kn = colors.to_rgba('tab:green', 1)
    fill_kn = colors.to_rgba('tab:green', 0.04)

    index = 0
    for protocol in args.protocols:
      for l, loss in enumerate(args.loss_functions):
        # Calculate histogram
        drop_bg = loss in ["garbage", "BG B", "BG 1", "BG F", "BG FK", "BG FN"]  #  Drop the background class
        if scores[protocol][loss] is not None:
          kn_hist_val, kn_edges_val, neg_hist_val, neg_edges_val = openset_imagenet.util.get_histogram(
              scores[protocol][loss]["val"],
              unk_label=-1,
              metric='score',
              bins=bins,
              drop_bg=drop_bg
          )
        else:
          kn_hist_val, kn_edges_val, neg_hist_val, neg_edges_val = [], [0], [], [0]

        # Plot histograms
        axs[index].stairs(kn_hist_val, kn_edges_val, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
        axs[index].stairs(neg_hist_val, neg_edges_val, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)

       # axs[ix].set_yscale('log')
        axs[index].set_title(f"$P_{{{protocol}}}$ {args.labels[l]}")
        index += 1

    # Share y axis of the histograms of the same protocol
    for p in range(P):
      for l in range(1,N):
        axs[N*p+l-1].sharey(axs[N*p+l])

    for ax in axs:
        # set the tick parameters for the current axis handler
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.label_outer()

    # Manual legend
    # For one protocol
    if len(args.protocols) == 1:
      axs[-2].legend(['Known', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(0.8, -1.6),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # For four protocols
    elif len(args.loss_functions) == 4:
      axs[-2].legend(['Known', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(-1, -0.05),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # For five protocols
    elif len(args.loss_functions) == 5:
      axs[-2].legend(['Known', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(-1, -0.05),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)

    else:
      axs[-2].legend(['Known', 'Negative'],
                    frameon=False,
                    fontsize=font_size-1,
                    bbox_to_anchor=(0.1, -0.08),
                    ncol=2,
                    handletextpad=0.3,
                    columnspacing=1,
                    markerscale=1)
    # X label
    fig.text(0.5, 0.02, 'Validation Set', ha='center', fontsize=font_size)



def conf_and_ccr_table(args, scores, epochs):

  def find_nearest(array, value):
      """Get the closest element in array to value"""
      array = numpy.asarray(array)
      idx = (numpy.abs(array - value)).argmin()
      return idx, array[idx]

  query = [1e-3, 1e-2, 0.1,1.0]
  unk_label = -2

  with open(args.table_path, "w") as table:
    for p, protocol in enumerate(args.protocols):
      for l, loss in enumerate(args.loss_functions):
        for which in ["test"]:
          array = scores[protocol][loss][which]
          gt = array['gt']
          values = array['scores']

          ccr_, fpr_ = openset_imagenet.util.calculate_oscr(gt, values, unk_label=unk_label)

          # get confidences on test set
          offset = 0 if loss in ["garbage", "BG B", "BG 1", "BG F", "BG FK", "BG FN"] else 1 / (numpy.max(gt)+1)
          last_valid_class = -1 if loss in ["garbage", "BG B", "BG 1", "BG F", "BG FK", "BG FN"] else None
          c = openset_imagenet.metrics.confidence(
              torch.tensor(values),
              torch.tensor(gt, dtype=torch.long),
              offset = offset, unknown_class=-2, last_valid_class=last_valid_class
          )


          # write loss and confidences
          table.write(f"$P_{protocol}$ - {args.labels[l]} & {epochs[protocol][loss][0]} & {c[0]:1.3f} & {c[2]:1.3f}")

          for q in query:
              idx, fpr = find_nearest(fpr_, q)
              error = round(100*abs(fpr - q) / q, 1)
              if error >= 10.0:  # If error greater than 10% then assume fpr value not in data
                  table.write(" & ---")
              else:
                  table.write(f" & {ccr_[idx]:1.3f}")
        table.write("\\\\\n")
      if p < len(args.protocols)-1:
        table.write("\\midrule\n")


def main():
  args = get_args()


  print("Extracting and loading scores")
  print("Protocol(s): ")
  for protocol in args.protocols:
    print(protocol)
  print("Losses: ")
  for loss in args.loss_functions:
    print(loss)
  scores, epoch = load_scores(args)

  print("Writing file", args.file_name)
  pdf = PdfPages(args.file_path)
  try:
    # plot OSCR (actually not required for best case)
    print("Plotting OSCR curves")
    plot_OSCR(args, scores)
    pdf.savefig(bbox_inches='tight', pad_inches = 0)

    # TODO: Give no argument for the confidence propagation plot
    if not args.linear and not args.sort_by_loss:
      # plot confidences
      print("Plotting confidence plots")
      plot_confidences(args)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)

      # plot histograms
      print("Plotting softmax testing histograms")
      plot_softmax(args, scores)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)

      print("Plotting softmax validation histograms")
      plot_softmax_validation(args, scores)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)

  finally:
    pdf.close()

  # create result table
  if not args.linear and not args.sort_by_loss:
    print("Creating Table")
    print("Writing file", args.table_path)
    conf_and_ccr_table(args, scores, epoch)
