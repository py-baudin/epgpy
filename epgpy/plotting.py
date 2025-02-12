"""EPG plots / sequence diagrams"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from . import operators, functions, statematrix


def show():
    """shortcut for plt.show"""
    plt.show()


def plot_epg(
    seq,
    *,
    kvalue=1,
    kgrid=None,
    yaxis=0,
    ops="S,T,E",
    title=None,
    figname=None,
    calpha=0.5,
    cwidth=0,
):
    """plot RF/gradients and EPG diagram"""

    # properties
    alpha_pow = calpha
    width_factor = 2
    width_pow = cwidth

    # flatten sequence
    seq = functions.flatten_sequence(seq)
    kdim = min(functions.getkdim(seq), 3)
    shape = functions.getshape(seq)
    index = (0,) * len(shape)

    # yaxis
    yaxis = np.arange(kdim)[yaxis]
    if kdim == 2:
        cx = {0: 1, 1: 0}[yaxis]
    elif kdim == 3:
        cx, cy = tuple(np.mod(np.arange(1, 3) + yaxis, 3))

    # max shift
    shift, kmin, kmax = 0, 0, 0
    for op in seq:
        if isinstance(op, operators.S):
            shift += get_shift(op, kvalue)[(0,) * op.ndim]
            kmax = np.maximum(kmax, shift)
            kmin = np.minimum(kmin, shift)
    kmax = np.maximum(kmax, np.abs(kmin))
    kmin = -kmax

    # operators to simulate
    ops = ops.split(",")

    # initial state matrix
    sm = statematrix.StateMatrix(kgrid=kgrid)

    # plotting
    fig = plt.figure(figname, figsize=(8, 6))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=(1, 1 * kdim, 8))

    # epg axis
    ax_epg = fig.add_subplot(gs[2, 0])
    # rf axis
    ax_rf = fig.add_subplot(gs[0, 0])
    # gradients axes
    gs_grad = gridspec.GridSpecFromSubplotSpec(kdim, 1, subplot_spec=gs[1, 0])
    ax_grad = [fig.add_subplot(gs_grad[i, 0]) for i in range(kdim)]

    # loop over operatorss
    times = [0]
    ymax = 0
    for op in seq:
        prev = times[-1]
        now = op.duration + prev
        times.append(now)

        if type(op).__name__ in ops:
            pass
        elif isinstance(op, operators.Probe):
            pass
        else:
            # bypass operators
            continue

        # apply operator
        sm = op(sm)

        Fs = sm.F[index[: sm.ndim]]
        Zs = sm.Z[index[: sm.ndim]]
        ks = sm.k[index[: sm.k.ndim - 2]]

        # plot operator
        if isinstance(op, operators.S):
            shift = get_shift(op, kvalue)[index[: op.ndim]]

            # gradient box
            for i in range(kdim):
                plt.sca(ax_grad[i])
                plt.fill_between([prev, now], [shift[i]] * 2, color="gray", alpha=0.3)

            # EPG plot
            plt.sca(ax_epg)
            for i in range(2 * sm.nstate + 1):
                fmag = np.minimum(np.abs(Fs[i]), 1)
                zmag = np.abs(Zs[i])
                kvals = np.stack([ks[i] - shift, ks[i]], axis=0)
                if kdim == 1:
                    yvals = kvals[:, 0]
                    fcolor, zcolor = "k", "k"
                elif kdim == 2:
                    yvals = kvals[:, yaxis]
                    meank = np.mean(kvals, axis=0)
                    fcolor = cm1d(meank[cx], kmax[cx])
                    zcolor = cm1d(kvals[1, cx], kmax[cx])
                else:
                    yvals = kvals[:, yaxis]
                    meank = np.mean(kvals, axis=0)
                    fcolor = cm2d(meank[cx], meank[cy], xmax=kmax[cx], ymax=kmax[cy])
                    zcolor = cm2d(
                        kvals[1, cx], kvals[1, cy], xmax=kmax[cx], ymax=kmax[cy]
                    )
                falpha = fmag**alpha_pow
                zalpha = zmag**alpha_pow
                flinewidth = width_factor * fmag**width_pow
                zlinewidth = width_factor * zmag**width_pow

                if i >= sm.nstate and zmag > 1e-5:
                    # z states
                    plt.plot(
                        [prev, now],
                        [yvals[1]] * 2,
                        ":",
                        color=zcolor,
                        lw=zlinewidth,
                        alpha=zalpha,
                    )

                if fmag > 1e-5:
                    # fstates
                    plt.plot(
                        [prev, now], yvals, color=fcolor, lw=flinewidth, alpha=falpha
                    )
                    ymax = max(np.max(np.abs(yvals)), ymax)

        if isinstance(op, operators.T):
            # plot RF pulse
            alpha = np.asarray(op.alpha).flat[0]
            phi = np.asarray(op.phi).flat[0]
            plt.sca(ax_rf)
            plt.vlines(now, 0, alpha, color="k")
            va = {False: "top", True: "bottom"}[alpha > 0]
            plt.annotate(
                f"{alpha:.0f}°", (now, alpha + 2 * np.sign(alpha)), va=va, ha="center"
            )
            if not np.isclose(phi, 0):
                plt.annotate(f"{phi:.0f}°", (now, 0), va="bottom")

            # plot line on EPG plot
            plt.sca(ax_epg)
            plt.scatter(
                now, [0], marker="o", color="gray", facecolors="white", zorder=10
            )
            plt.axvline(now, linestyle=":", color="gray", alpha=0.5)

        if isinstance(op, operators.Probe):
            plt.sca(ax_epg)
            plt.scatter(now, [0], marker="v", color="gray", zorder=10)

    # epg
    plt.sca(ax_epg)
    straxes = {0: "kx", 1: "ky", 2: "kz"}
    xlim = times[0] - 3e-2 * times[-1], times[-1] * (1 + 3e-2)
    plt.xlim(xlim)
    ymin = -ymax
    ptp = ymin - ymax
    plt.ylim(ymin - 0.05 * ptp, ymax + 0.05 * ptp)
    plt.ylabel(straxes[yaxis])
    plt.xlabel("time (ms)")
    plt.axhline(0, color="k", zorder=-1)
    if kdim == 2:
        colorbar1d(xmax=kmax[cx], x=straxes[cx])
    elif kdim == 3:
        x, y = straxes[cx], straxes[cy]
        colorbar2d(xmax=kmax[cx], ymax=kmax[cy], x=x, y=y)

    # rf pulses
    plt.sca(ax_rf)
    plt.xlim(xlim)
    plt.ylim([np.sign(y) * 180 for y in plt.ylim()])
    plt.annotate(
        "Rf",
        xy=(-1e-2, 0.5),
        ha="right",
        va="center",
        xycoords="axes fraction",
        weight="bold",
    )
    plt.axhline(0, color="k")
    plt.axis("off")

    # gradient axes
    for k in range(kdim):
        axname = {0: "x", 1: "y", 2: "z"}[k]
        plt.sca(ax_grad[k])
        plt.axhline(0, color="k")
        plt.annotate(
            f"G{axname}",
            xy=(-1e-2, 0.5),
            ha="right",
            va="center",
            xycoords="axes fraction",
            weight="bold",
        )
        # plt.annotate("", xy=(1, 0), xytext=(0, 0), xycoords=coords, textcoords=coords, arrowprops=dict(arrowstyle="-|>"))
        plt.xlim(xlim)
        plt.axis("off")
    # plt.annotate('t', xy=(1+1e-2,0), ha='left', va='center', xycoords='axes fraction', weight='bold')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    return fig


def colorbar1d(xmax, *, x="", rect=[0.03, 0.7, 0.1, 0.2], n=11, fig=None):
    ax = plt.gca()
    inset = ax.inset_axes(rect)
    coords = np.linspace(-1, 1, n)[:, np.newaxis]
    img = cm1d(coords, 1)
    # breakpoint()
    inset.imshow(img, origin="lower")
    inset.set_xticks([])
    inset.set_yticks([0, (n - 1) / 2, n - 1])
    inset.set_yticklabels((f"{v:.1f}" for v in [-xmax, 0, xmax]), fontsize=8)
    inset.annotate(x, xy=(0, 1.02), ha="right", va="bottom", xycoords="axes fraction")


def colorbar2d(
    xmax, ymax, *, x="", y="", rect=[0.03, 0.75, 0.15, 0.15], n=11, fig=None
):
    """add 2d colorbar to axis"""
    ax = plt.gca()
    inset = ax.inset_axes(rect)
    coords = 2 * np.stack(np.indices((n, n)), axis=-1) / (n - 1) - 1
    img = cm2d(coords[..., 0], coords[..., 1], 1, 1)

    inset.imshow(img, origin="lower")
    inset.set_xticks([0, (n - 1) / 2, n - 1])
    inset.set_yticks([0, (n - 1) / 2, n - 1])
    inset.set_xticklabels(
        (f"{v:.1f}" for v in [-ymax, 0, ymax]), fontsize=8, rotation=90
    )
    inset.set_yticklabels((f"{v:.1f}" for v in [-xmax, 0, xmax]), fontsize=8)
    inset.annotate(y, xy=(1.2, 0), ha="left", va="top", xycoords="axes fraction")
    inset.annotate(x, xy=(0, 1.02), ha="right", va="bottom", xycoords="axes fraction")


def cm1d(x, xmax):
    x = np.array(x)
    if xmax:
        x = (np.clip(x, -xmax, xmax) + xmax) / (2 * xmax)
    return plt.cm.plasma(x)


def cm2d(x, y, xmax, ymax, pow=0.5):
    """2d color map"""
    # breakpoint()
    x = np.array(x)
    y = np.array(y)
    if xmax:
        x = 2 * (np.clip(x, -xmax, xmax) + xmax) / (2 * xmax) - 1
    if ymax:
        y = 2 * (np.clip(y, -ymax, ymax) + ymax) / (2 * ymax) - 1

    colors = np.zeros(x.shape + (3,))
    # x: green-red
    yax = np.minimum(-y + 1, 1)
    colors[..., 0] = np.clip(x + yax, 0, 1)
    colors[..., 1] = np.clip(-x + yax, 0, 1)
    # y: yellow-blue
    colors[..., 2] = np.clip(y + 1, 0, 1)

    return (1 - colors) ** pow


def get_shift(op, kvalue):
    if not isinstance(op, operators.S):
        return 0
    return np.atleast_2d(op.k) * kvalue
