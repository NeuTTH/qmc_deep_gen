"""Single source of colour for every QLVM figure in this subproject.

The rule is one variable -> one colour or colormap, in every figure that draws it.
Before this module the same four session types were drawn in one palette by the
dataset-construction figures (``scripts/dataset_construct/0*_fig*.py``) and in a
different one by the inference figures, emitter sex reused the session-type pink
for a different meaning, and spectrograms were drawn in ``viridis`` in the static
panels but ``magma`` in the videos.

The categorical palette here is the one the dataset-construction figures already
use (seaborn ``deep``), so a per-run figure and fig1-fig9 can be laid side by side.
Import from here rather than re-declaring hexes locally.
"""

# --------------------------------------------------------------------------- #
# Categorical: session type / recording condition
# --------------------------------------------------------------------------- #
# Canonical short codes, matching the ``session_type`` column that
# build-qlvm-training-set writes.
SESSION_TYPE_COLORS = {
    "MF": "#4C72B0",
    "FF": "#C44E52",
    "MM": "#55A868",
    "lone_male": "#8172B2",
    # Condition refinements of a sex-derived pairing, produced by
    # analyze_mouse_latents_2d._refine_session_types when a session_id,condition
    # table is supplied. build-qlvm-training-set derives session_type from subject
    # sex alone, so mute-female courtship sessions are 'MF' next to the intact ones.
    # MF_intact keeps the MF blue so anything already drawn reads unchanged, and
    # MF_mute is a darker shade of the SAME hue -- the two are variants of one
    # pairing, not two unrelated conditions, and the palette should say so.
    "MF_intact": "#4C72B0",
    "MF_mute": "#2A4A7F",
}
SESSION_TYPE_ORDER = ["MF", "MF_intact", "MF_mute", "FF", "MM", "lone_male"]

# The long labels the legacy spec_id parse produces, mapped onto the same hexes so
# a run analysed from either kind of dataset comes out the same colour.
COND_ORDER = ["Male-Female", "Female-Female", "Male-Male", "Lone-Male"]
COND_COLORS = {
    "Male-Female": SESSION_TYPE_COLORS["MF"],
    "Female-Female": SESSION_TYPE_COLORS["FF"],
    "Male-Male": SESSION_TYPE_COLORS["MM"],
    "Lone-Male": SESSION_TYPE_COLORS["lone_male"],
}

# Aliases seen in the wild: the dataset-construction figures spell the mixed-sex
# condition ``MF_intact``, and the long labels appear wherever spec_ids were parsed.
_SESSION_TYPE_ALIASES = {
    "MF_intact": "MF",
    "Male-Female": "MF",
    "Female-Female": "FF",
    "Male-Male": "MM",
    "Lone-Male": "lone_male",
    "lone-male": "lone_male",
}

# Anything unrecognised, and every sample whose label is missing.
MISSING_COLOR = "#BFBFBF"
UNKNOWN_COLOR = "#8C8C8C"


def session_type_color(label):
    """Colour for a session type / condition label, under any of its spellings."""
    key = _SESSION_TYPE_ALIASES.get(str(label), str(label))
    return SESSION_TYPE_COLORS.get(key, UNKNOWN_COLOR)


# --------------------------------------------------------------------------- #
# Categorical: emitter sex
# --------------------------------------------------------------------------- #
# Deliberately outside the session-type palette. These used to be drawn in the
# same pink and blue the session types used, which made a sex panel and a type
# panel look like the same variable.
EMITTER_SEX_COLORS = {"female": "#DA8BC3", "male": "#64B5CD"}
EMITTER_SEX_ORDER = ["female", "male"]


# --------------------------------------------------------------------------- #
# Continuous: one colormap per quantity
# --------------------------------------------------------------------------- #
CMAP_SPEC = "viridis"      # every spectrogram image, static or animated
CMAP_POSTERIOR = "magma"   # aggregated-posterior density, every panel that shows it
CMAP_FREQ = "viridis"      # mean frequency
CMAP_BANDWIDTH = "plasma"  # spectral bandwidth -- deliberately not CMAP_FREQ, so the
                           # two frequency-axis panels are not mistaken for each other
CMAP_MASK_COUNT = "cividis"  # SAM mask count
CMAP_DURATION = "magma"    # syllable duration
CMAP_SOCIAL_DIST = "YlOrRd_r"  # social distance, near = light
CMAP_SEGMENT = "RdYlGn"    # segment trajectories, coloured by social distance

# Overlay ink on top of a dark density image (watershed boundaries, centroids).
OVERLAY_COLOR = "white"
HIGHLIGHT_COLOR = "#00E5FF"

# Fixed range for social distance so every panel and every run shares a scale.
SOCIAL_DIST_RANGE = (0, 85)

# Fixed range for the SAM mask count colour scale. The distribution is heavily
# skewed -- almost every syllable carries 1-3 masks and a thin tail runs past 20 --
# so an autoscaled colorbar spends its whole range on samples that do not exist and
# renders the bulk of the data as one flat colour. Clipping at 8 puts the contrast
# where the samples are. Points above 8 are still drawn, saturated at the top
# colour, never dropped.
MASK_COUNT_RANGE = (1, 8)

# The high-mask-count panel: the counts that get their own discrete colour, and
# everything outside that set drawn in MISSING_COLOR behind them. Okabe-Ito hexes,
# chosen to be colourblind-safe and to sit outside the seaborn-deep session-type
# palette, so a high-mask panel is never read as a condition panel.
HIGH_MASK_COUNTS = [4, 5, 6, 7]
HIGH_MASK_COLORS = {
    "4": "#E69F00",
    "5": "#009E73",
    "6": "#0072B2",
    "7": "#CC79A7",
}


def bar_shades(cmap_name, n, lo=0.15, hi=0.85):
    """`n` colours sampled across a colormap, for bars over an ordered variable.

    Lets a bar chart of mean MSE by duration bin carry the same colour ramp as the
    latent scatter coloured by duration, instead of an unrelated flat colour.
    """
    import matplotlib as mpl
    import numpy as np

    cmap = mpl.colormaps[cmap_name]
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(v) for v in np.linspace(lo, hi, n)]


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
# Train and val appear on one axis in the training-stats figure, and nowhere else,
# so they take two hues that do not collide with the session-type palette.
TRAIN_COLOR = "#4C72B0"
VAL_COLOR = "#DD8452"


def scatter_params_for(n_points, size, alpha, reference=150_000):
    """Scale marker size and alpha so a scatter reads at any sample count.

    The defaults are tuned for a full embedding (~150k points), where small and
    nearly transparent is what keeps the structure visible. A subsampled run drawn
    with the same numbers comes out as a blank square. Runs at or above the
    reference count are returned unchanged, so production figures do not move.
    """
    if n_points <= 0 or n_points >= reference:
        return size, alpha
    f = (reference / n_points) ** 0.5
    return min(size * min(f, 3.0), 40.0), min(alpha * min(f, 4.0), 0.9)
