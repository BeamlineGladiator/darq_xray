# Estimator recalibration — measurement campaign (2026-08-26)

Box: this machine. Harness: `recal.py` (synthetic fixtures) + `tests/peak_rss.py`.
Every figure is **child peak RSS**, sampled at 20 ms. Runs were serial.

## Process floors (`measure_process_floor`, data ~ noise)

| stage | floor MiB |
|---|---|
| strain, save_plots on | 166.2 |
| strain, save_plots off | 96.7 |
| mosaicity | 74.1 |
| rocking | 154.3 |
| matched | 156.6 |

## strain — peak does NOT scale with n_layers, but is not flat either

1266x1832, plots on: n=1 294.6 | n=3 393.2 | n=8 465.2 | n=16 466.2 MiB.
Plots **off** is flat in n (108.9 at n=3 vs 108.5 at n=8) — the growth is entirely
matplotlib's, and it saturates by n≈8. All shape sweeps below are at n=8.

| layer elems | plots off MiB | plots on MiB |
|---|---|---|
| 0.262 M (512x512)   | 108.5 | 312.6 |
| 0.590 M (768x768)   | 122.8 | 350.4 |
| 1.049 M (1024x1024) | 145.7 | 392.8 |
| 2.319 M (1266x1832) | 202.5 | 465.1 |
| 3.520 M (1600x2200) | 257.3 | 571.4 |
| 5.243 M (2048x2560) | 307.8 | 787.3 |

Fits: **off** 43.0 B/elem + 102.0 MiB (r2 0.992, local slopes 30.7-52.3) — the
intercept lands on the independently measured 96.7 MiB floor.
**on** 94.7 B/elem + 284.5 MiB (r2 0.981, local 59.7-131.4).
Plotting marginal = 51.7 B/elem + 182.5 MiB.

### The plotting term is NOT data-independent (the handoff assumed it was)

Half of it scales with the data (four `imshow` rasterisations: float64 norm +
uint8 RGBA per full map). The other half scales with the **canvas**, which a
fixed-scale style can inflate by 6x. 1266x1832, n=8, plots on:

| style | figsize in | canvas Mpx | measured MiB |
|---|---|---|---|
| none (legacy 7 x 7*aspect+1.5) | 7 x 13.86 | 5.61 | 465.1 |
| scale_um_per_cm=20 | 6.97 x 11.16 | 4.84 | 579.3 |
| scale_um_per_cm=10 | 12.44 x 20.82 | 12.09 | 971.2 |
| scale_um_per_cm=6.4 (30-in clamp) | 18.6 x 31.5 | 25.03 | 1682.2 |

Local slopes between consecutive fixed-scale points: **54.1, 54.4, 54.2 B per
canvas pixel** — the most linear relationship measured in this campaign. The
box path costs more per pixel than the legacy path because `fit_axes_to_box`
re-renders; charging the box rate for both is the safe direction.

## mosaicity — flat in n_layers, exactly one layer per dataset

1266x1832: n=1 176.1 | n=4 194.1 | n=8 194.1 | n=16 194.1 MiB.

| layer elems | ds=1 | ds=2 | ds=3 | ds=4 |
|---|---|---|---|---|
| 2.319 M | 123.3 | 141.0 | 176.1 | 194.1 |
| 5.243 M | 184.8 | 224.8 | 265.2 | 305.4 |

Per-dataset slope at 5.243 M: **1.00, 1.01, 1.00 float64 layers** — exact.
Dataset-independent remainder: 1.77 layers (the gzip chunk buffer + read buffer).
Shape sweep at ds=4: 87.9 / 104.7 / 128.5 / 194.1 / 255.8 / 305.4 MiB over the
same six shapes; slope 45.8 B/elem, intercept 76.4 MiB against a measured 74.1.

## rocking — the estimator UNDER-predicts everywhere (2.4x to 16x)

save_layers on, animation/topview off unless stated.

| config | measured MiB |
|---|---|
| 256x256 f21 s6 | 179.0 |
| 384x384 f21 s6 | 190.6 |
| 512x512 f21 s6 | 211.1 |
| 724x724 f21 s6 | 292.0 |
| 1024x1024 f21 s6 | 423.9 |
| 512x512 f21 s3 / s12 / s24 | 192.3 / 295.7 / 442.4 |
| 512x512 s6 f51 / f101 / f201 | 211.1 / 262.2 / 408.3 |

Two-regressor fit (r2 **0.997**): the per-scan read is not the peak at f=21 —
**48.5 B per (scan x layer-elem)** + 156.2 MiB explains everything. The scan read
only bites above f~51, where its local slope is **5.85 B/scan-elem** = exactly
`itemsize + 4` (uint16 + the float32 copy).

### `save_topview` costs a data-independent ~365 MiB and is default ON

| volume | topview off | topview on | delta |
|---|---|---|---|
| 256x256 s6 | 180.0 | 541.4 | 361.4 |
| 512x512 s6 | 211.0 | 584.9 | 373.9 |
| 724x724 s6 | 292.1 | 646.5 | 354.4 |
| 1024x1024 s6 | 423.6 | 777.1 | 353.5 |
| 512x512 s12 | 291.6 | 666.6 | 375.0 |

It is the VTK/pyvista import plus a software GL context (EGL fails on this box),
not the data. `save_animation` costs +8 MiB, `save_layers` +34 MiB — both fold
into the base. With every default on, the shipped estimator was **14x under**.

## matched — flat in frames (the blocking works), scales with the FRAME

| frame elems | measured MiB |
|---|---|
| 0.0655 M (256x256) | 176.3 |
| 0.262 M (512x512) | 217.6 |
| 0.524 M (724x724) | 238.7 |
| 1.049 M (1024x1024) | 287.3 |
| 2.096 M (1448x1448) | 390.0 |

512x512 at f=21/51/101/201: 217.6 / 213.5 / 217.8 / 215.8 — **flat**, because
`load_pco_ff_frame` bounds the median's working set at
`MEDIAN_BLOCK_WORKING_SET_BYTES` (128 MiB). The shipped model's
`scan_elems * (itemsize + 16)` grows without bound with the frame count and so
over-predicts 4.3x at f=201 while its per-frame term is right: local slopes
84-103 B/frame-elem against the documented `12 * frame_elems * 8` = 96.


---

## Real STO2 validation (2026-08-26, after the rewrite)

| stage | configuration | model | measured | model/measured |
|---|---|---|---|---|
| strain | 76 layers, 1266x1832, plots on | 0.773 GiB | 0.494 GiB | **1.56** |
| mosaicity | 76 layers, four datasets | 0.215 GiB | 0.182 GiB | **1.18** |
| rocking | 10 mosa layers, 575 frames, ROI 700x1832 | 4.906 GiB | 4.290 GiB | **1.14** |
| matched | 76 matched layers, 50 frames, 2048^2 detector | 1.842 GiB | 1.582 GiB | **1.16** |
| rocking | all 76 mosa layers, 575 frames, ROI 700x1832 | 11.168 GiB | 5.841 GiB | **1.91** |

Before the rewrite, on the same data: strain **5.2x over** (2.627 vs 0.508),
mosaicity **36x over** (6.566 vs 0.181).

### rocking's old model was wrong in both directions

No single correction factor would have fixed it:

| configuration | old model | measured | old error |
|---|---|---|---|
| synthetic 512x512, f21, s6, default toggles | 43.5 MiB | 584.9 MiB | **13.4x UNDER** |
| real STO2, 10 mosa layers (575 frames, 2048^2 detector, ROI 700x1832) | 13.789 GiB | 4.290 GiB | 3.2x over |

It under-charged the volume term (20 B/elem/scan against a measured 48.5) and
charged nothing for `save_topview`, but it also sized the per-scan read from the
whole detector when `process_raw_scan` reads only the ROI. On short scans the
under-charges won; on 575-frame scans the inflated read term hid them.


---

## Second round: what the real run caught that the synthetic sweep could not

The first recalibration over-predicted every synthetic point and then came in at
**0.43x** of the real `matched` measurement. Two regimes the fixtures never
entered:

1. **The colour-limit pool.** `run()` pools up to **10** whole frames,
   `np.concatenate`s them while `pooled` is still live, and `np.percentile`
   partitions a copy of *that* — three pool-sized arrays. A four-folder fixture
   caps the pool at four. Re-measured with **fourteen** folders, auto minus
   fixed clim:

   | frame elems | auto MiB | fixed MiB | delta | delta / frame |
   |---|---|---|---|---|
   | 0.0655 M | 195.4 | 189.7 | 5.7 | 11.1 frames |
   | 0.262 M | 272.4 | 249.2 | 23.2 | 11.6 |
   | 0.524 M | 321.4 | 267.6 | 53.8 | 13.4 |
   | 1.049 M | 422.3 | 335.1 | 87.2 | 10.9 |
   | 2.097 M | 616.8 | 422.1 | 194.7 | 12.2 |

2. **The samy X-padding.** `_apply_shift_single` builds an `(ny, nx + pads)`
   canvas per layer. Isolated at 1024x1024 by spreading samy:

   | pad | padded elems | measured MiB | B per extra element |
   |---|---|---|---|
   | 0 | 1.049 M | 412.4 | — |
   | 0.5x | 1.573 M | 463.8 | 98 |
   | 1x | 2.097 M | 525.2 | 108 |
   | 2x | 3.146 M | 631.8 | 105 |

   On real STO2 the pad is **1058 px on a 2048-wide detector** — 52%.

A code review then found the same class of miss in `rocking`: it priced the
frames it reads, not the volume that accumulates. `apply_samy_shifts_to_volume`
and `interpolate_to_uniform_z` inflate that, and on real STO2 the aligned volume
is **(76, 700, 2891)** against a 1832-column read — 1.58x. The synthetic fixture
had uniform samz (so `n_uniform == n` identically) and a samy spread giving a
1.06-1.30x pad, so neither inflation was exercised.

**The lesson, in one line: a fixture that does not enter the regime a term
describes cannot calibrate that term, and over-predicting everywhere is not
evidence that it does.**


## rocking's two regimes (why the 76-layer ratio is 1.91 and not 1.14)

`ROCKING_VOLUME_BYTES_PER_ELEM = 48` was fitted on 21-frame synthetic scans,
where the accumulated volumes ARE the peak. Backing the floor and the per-scan
read out of the two real measurements gives the volume's own marginal on real
data:

| run | floor + read | measured | implied volume cost | aligned elems | B per aligned elem |
|---|---|---|---|---|---|
| 10 layers | 4397 MiB | 4393 MiB | ~0 | 13.7 M | ~0 |
| 76 layers | 4397 MiB | 5981 MiB | 1584 MiB | 153.8 M | **~11** |

So on 575-frame scans the read dominates and 48 over-charges by ~4x on that
term. The model keeps 48 because the small-frame regime really does measure it
and a model must cover both; the sum (rather than a max) is kept because the
per-scan read and the accumulated volumes genuinely coexist.

**A prediction that the measurement refuted.** The code review predicted this
run would come in UNDER (~0.84) once the alignment inflation was priced. It came
in at 1.91 over — and the pre-fix model would have been 1.48 over, not under.
The alignment fix is still right (it is the safe direction and the inflation is
real: (76, 700, 2891) against a 1832-column read), but the reasoning that
predicted an under-prediction assumed the aligned volume costs what the
synthetic fixtures said, which on real frame counts it does not.
