from multiprocessing.pool import Pool

from scipy.stats import norm,rankdata
from numpy.random import seed,shuffle,choice, get_state, set_state
from numpy import diff, full, nan, absolute, in1d, asarray, round, array_split, concatenate, apply_along_axis,isinf, isnan,sqrt,where,nanmax, nanmin, unique, exp, finfo, log, log2, sign, power, e, pi

from pandas import concat, DataFrame, Series
from math import ceil
from warnings import warn

from statsmodels.sandbox.stats.multicomp import multipletests

import sys

from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from seaborn import husl_palette

from scipy.cluster.hierarchy import dendrogram, linkage

from plotly.offline import iplot
from plotly.offline import plot as offline_plot
# from plotly.plotly import plot as plotly_plot
from chart_studio.plotly import plot as plotly_plot


#from rpy2.robjects.numpy2ri import numpy2ri
# import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri

#import numpy as np
from scipy.stats import pearsonr

eps = finfo(float).eps

# ro.conversion.py2ri = numpy2ri

pandas2ri.activate()

mass = importr("MASS")

def _make_annotations(score_moe_p_value_fdr):

    annotations = DataFrame(index=score_moe_p_value_fdr.index)

    if score_moe_p_value_fdr["0.95 MoE"].isna().all():

        annotations["Score"] = score_moe_p_value_fdr["Score"].apply("{:.2f}".format)

    else:

        annotations["Score(\u0394)"] = score_moe_p_value_fdr[
            ["Score", "0.95 MoE"]
        ].apply(lambda score_moe: "{:.2f}({:.2f})".format(*score_moe), axis=1)

    if not score_moe_p_value_fdr["P-Value"].isna().all():

        function = "{:.2e}".format

        annotations["P-Value"] = score_moe_p_value_fdr["P-Value"].apply(function)

        annotations["FDR"] = score_moe_p_value_fdr["FDR"].apply(function)

    return annotations
def compute_nd_array_margin_of_error(nd_array, confidence=0.95, raise_for_bad=True):

    is_good = ~check_nd_array_for_bad(nd_array, raise_for_bad=raise_for_bad)

    if is_good.any():

        nd_array_good = nd_array[is_good]

        return norm.ppf(q=confidence) * nd_array_good.std() / sqrt(nd_array_good.size)

    else:

        return nan

def _match_randomly_sampled_target_and_data_to_compute_margin_of_errors(
    target,
    data,
    random_seed,
    n_sampling,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    print("Computing MoE with {} sampling ...".format(n_sampling))

    seed(random_seed)

    index_x_sampling = full((data.shape[0], n_sampling), nan)

    n_sample = ceil(0.632 * target.size)

    if n_sampling == 0:
        return([0] * data.shape[0])
    
    for i in range(n_sampling):

        random_indices = choice(target.size, size=n_sample, replace=True)

        sampled_target = target[random_indices]

        sampled_data = data[:, random_indices]

        random_state = get_state()

        index_x_sampling[:, i] = _match_target_and_data(
            sampled_target,
            sampled_data,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    return apply_along_axis(
        compute_nd_array_margin_of_error, 1, index_x_sampling, raise_for_bad=False
    )



def check_nd_array_for_bad(nd_array, raise_for_bad=True):

    bads = []

    is_nan = isnan(nd_array)

    if is_nan.any():

        bads.append("nan")

    is_inf = isinf(nd_array)

    if is_inf.any():

        bads.append("inf")

    is_bad = is_nan | is_inf

    n_bad = is_bad.sum()

    if 0 < n_bad:

        message = "{} good & {} bad ({}).".format(
            nd_array.size - n_bad, n_bad, ", ".join(bads)
        )

        if raise_for_bad:

            raise ValueError(message)

        else:

            warn(message)

    return is_bad
    
def compute_empirical_p_value(
    value, random_values, p_value_direction, raise_for_bad=True
):

    if isnan(value):

        return nan

    is_good = ~check_nd_array_for_bad(random_values, raise_for_bad=raise_for_bad)

    if is_good.any():

        random_values_good = random_values[is_good]

        if p_value_direction == "<":

            n_significant_random_value = (random_values_good <= value).sum()

        elif p_value_direction == ">":

            n_significant_random_value = (value <= random_values_good).sum()

        return max(1, n_significant_random_value) / random_values_good.size

    else:

        return nan


def apply_function_on_2_1d_arrays(
    _1d_array_0,
    _1d_array_1,
    function,
    n_required=None,
    raise_for_n_less_than_required=True,
    n_permutation=0,
    random_seed=20121020,
    p_value_direction=None,
    raise_for_bad=True,
    use_only_good=True,
):

    is_good_0 = ~check_nd_array_for_bad(_1d_array_0, raise_for_bad=raise_for_bad)

    is_good_1 = ~check_nd_array_for_bad(_1d_array_1, raise_for_bad=raise_for_bad)

    if use_only_good:

        is_good = is_good_0 & is_good_1

        if n_required is not None:

            if n_required <= 1:

                n_required *= is_good.size

            if is_good.sum() < n_required:

                message = "{} requires {} <= n.".format(function.__name__, n_required)

                if raise_for_n_less_than_required:

                    raise ValueError(message)

                else:

                    warn(message)

                    return nan

        _1d_array_good_0 = _1d_array_0[is_good]

        _1d_array_good_1 = _1d_array_1[is_good]

    else:

        _1d_array_good_0 = _1d_array_0[is_good_0]

        _1d_array_good_1 = _1d_array_1[is_good_1]

    value = function(_1d_array_good_0, _1d_array_good_1)

    if 0 < n_permutation:

        random_values = full(n_permutation, nan)

        _1d_array_good_0_shuffled = _1d_array_good_0.copy()

        seed(random_seed)

        for i in range(n_permutation):

            shuffle(_1d_array_good_0_shuffled)

            random_values[i] = function(_1d_array_good_0_shuffled, _1d_array_good_1)

        return (
            value,
            compute_empirical_p_value(
                value, random_values, p_value_direction, raise_for_bad=raise_for_bad
            ),
        )

    else:

        return value


def _match_target_and_data(
    target,
    data,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    return apply_along_axis(
        apply_function_on_2_1d_arrays,
        1,
        data,
        target,
        match_function,
        n_required=n_required_for_match_function,
        raise_for_n_less_than_required=raise_for_n_less_than_required,
        raise_for_bad=False,
    )
    



def _permute_target_and_match_target_and_data(
    target,
    data,
    random_seed,
    n_permutation,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    print("Computing p-value and FDR with {} permutation ...".format(n_permutation))

    seed(random_seed)

    index_x_permutation = full((data.shape[0], n_permutation), nan)

    permuted_target = target.copy()

    for i in range(n_permutation):

        shuffle(permuted_target)

        random_state = get_state()

        index_x_permutation[:, i] = _match_target_and_data(
            permuted_target,
            data,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    return index_x_permutation
    




def compute_empirical_p_values_and_fdrs(
    values, random_values, p_value_direction, raise_for_bad=True
):

    is_good = ~check_nd_array_for_bad(values, raise_for_bad=raise_for_bad)

    is_good_random_value = ~check_nd_array_for_bad(
        random_values, raise_for_bad=raise_for_bad
    )

    p_values = full(values.shape, nan)

    fdrs = full(values.shape, nan)

    if is_good.any() and is_good_random_value.any():

        values_good = values[is_good]

        random_values_good = random_values[is_good_random_value]

        if "<" in p_value_direction:

            good_p_values_less = asarray(
                tuple(
                    compute_empirical_p_value(value_good, random_values_good, "<")
                    for value_good in values_good
                )
            )

            good_fdrs_less = multipletests(good_p_values_less, method="fdr_bh")[1]

        if ">" in p_value_direction:

            good_p_values_great = asarray(
                tuple(
                    compute_empirical_p_value(value_good, random_values_good, ">")
                    for value_good in values_good
                )
            )

            good_fdrs_great = multipletests(good_p_values_great, method="fdr_bh")[1]

        if p_value_direction == "<>":

            good_p_values = where(
                good_p_values_less < good_p_values_great,
                good_p_values_less,
                good_p_values_great,
            )

            good_fdrs = where(
                good_fdrs_less < good_fdrs_great, good_fdrs_less, good_fdrs_great
            )

        elif p_value_direction == "<":

            good_p_values = good_p_values_less

            good_fdrs = good_fdrs_less

        elif p_value_direction == ">":

            good_p_values = good_p_values_great

            good_fdrs = good_fdrs_great

        p_values[is_good] = good_p_values

        fdrs[is_good] = good_fdrs

    return p_values, fdrs

def select_series_indices(
    series,
    direction,
    threshold=None,
    n=None,
    fraction=None,
    standard_deviation=None,
    plot=True,
    title=None,
    xaxis=None,
    yaxis=None,
    html_file_path=None,
    plotly_file_path=None,
):

    series_sorted = series.dropna().sort_values()

    if n is not None:

        if direction in ("<", ">"):

            n = min(n, series_sorted.size)

        elif direction == "<>":

            n = min(n, series_sorted.size // 2)

    if fraction is not None:

        if direction in ("<", ">"):

            fraction = min(fraction, 1)

        elif direction == "<>":

            fraction = min(fraction, 1 / 2)

    if direction == "<":

        if threshold is None:

            if n is not None:

                threshold = series_sorted.iloc[n]

            elif fraction is not None:

                threshold = series_sorted.quantile(fraction)

            elif standard_deviation is not None:

                threshold = (
                    series_sorted.mean() - series_sorted.std() * standard_deviation
                )

        is_selected = series_sorted <= threshold

    elif direction == ">":

        if threshold is None:

            if n is not None:

                threshold = series_sorted.iloc[-n]

            elif fraction is not None:

                threshold = series_sorted.quantile(1 - fraction)

            elif standard_deviation is not None:

                threshold = (
                    series_sorted.mean() + series_sorted.std() * standard_deviation
                )

        is_selected = threshold <= series_sorted

    elif direction == "<>":

        if n is not None:

            threshold_low = series_sorted.iloc[n]

            threshold_high = series_sorted.iloc[-n]

        elif fraction is not None:

            threshold_low = series_sorted.quantile(fraction)

            threshold_high = series_sorted.quantile(1 - fraction)

        elif standard_deviation is not None:

            threshold_low = (
                series_sorted.mean() - series_sorted.std() * standard_deviation
            )

            threshold_high = (
                series_sorted.mean() + series_sorted.std() * standard_deviation
            )

        is_selected = (series_sorted <= threshold_low) | (
            threshold_high <= series_sorted
        )

    if plot:

        plot_and_save(
            dict(
                layout=dict(title=title, xaxis=xaxis, yaxis=yaxis),
                data=[
                    dict(
                        type="scatter",
                        name="All",
                        x=tuple(range(series_sorted.size)),
                        y=series_sorted,
                        text=series_sorted.index,
                        mode="markers",
                        marker=dict(color="#20d9ba"),
                    ),
                    dict(
                        type="scatter",
                        name="Selected",
                        x=is_selected.nonzero()[0],
                        y=series_sorted[is_selected],
                        text=series_sorted.index[is_selected],
                        mode="markers",
                        marker=dict(color="#9017e6"),
                    ),
                ],
            ),
            html_file_path,
            plotly_file_path,
        )

    return series_sorted.index[is_selected]


def _match(
    target,
    data,
    n_job,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
    n_extreme,
    fraction_extreme,
    random_seed,
    n_sampling,
    n_permutation,
):

    score_moe_p_value_fdr = DataFrame(columns=("Score", "0.95 MoE", "P-Value", "FDR"))

    n_job = min(data.shape[0], n_job)

    print(
        "Computing score using {} with {} process ...".format(
            match_function.__name__, n_job
        )
    )

    data_split = array_split(data, n_job)

    score_moe_p_value_fdr["Score"] = concatenate(
        multiprocess(
            _match_target_and_data,
            (
                (
                    target,
                    data_,
                    match_function,
                    n_required_for_match_function,
                    raise_for_n_less_than_required,
                )
                for data_ in data_split
            ),
            n_job,
        )
    )

    indices = select_series_indices(
        score_moe_p_value_fdr["Score"],
        "<>",
        n=n_extreme,
        fraction=fraction_extreme,
        plot=False,
    )

    score_moe_p_value_fdr.loc[
        indices, "0.95 MoE"
    ] = _match_randomly_sampled_target_and_data_to_compute_margin_of_errors(
        target,
        data[indices],
        random_seed,
        n_sampling,
        match_function,
        n_required_for_match_function,
        raise_for_n_less_than_required,
    )

    #print("Done Computing MoE...")
    
    p_values, fdrs = compute_empirical_p_values_and_fdrs(
        score_moe_p_value_fdr["Score"],
        concatenate(
            multiprocess(
                _permute_target_and_match_target_and_data,
                (
                    (
                        target,
                        data_,
                        random_seed,
                        n_permutation,
                        match_function,
                        n_required_for_match_function,
                        raise_for_n_less_than_required,
                    )
                    for data_ in data_split
                ),
                n_job,
            )
        ).flatten(),
        "<>",
        raise_for_bad=False,
    )

    score_moe_p_value_fdr["P-Value"] = p_values

    score_moe_p_value_fdr["FDR"] = fdrs

    return score_moe_p_value_fdr
    


COLOR_CATEGORICAL = (
    "#20d9ba",
    "#9017e6",
    "#ff1968",
    "#ffe119",
    "#3cb44b",
    "#4e41d8",
    "#ffa400",
    "#aaffc3",
    "#800000",
    "#e6beff",
    "#fffac8",
    "#0082c8",
    "#e6194b",
    "#006442",
    "#46f0f0",
    "#bda928",
    "#c91f37",
    "#fabebe",
    "#d2f53c",
    "#aa6e28",
    "#ff0000",
    "#808000",
    "#003171",
    "#ff4e20",
    "#a4345d",
    "#ffd8b1",
    "#bb7796",
    "#f032e6",
)
COLOR_WHITE_BLACK = ("#ebf6f7", "#171412")


def get_colormap_colors(colormap):

    if isinstance(colormap, str):

        colormap = get_cmap(colormap)

    return tuple(to_hex(colormap(i / (colormap.N - 1))) for i in range(colormap.N))
    




def make_categorical_colors(n_category):

    return tuple(to_hex(rgb) for rgb in husl_palette(n_colors=n_category))
def make_colorscale_from_colors(colors):

    if len(colors) == 1:

        colors *= 2

    return tuple((i / (len(colors) - 1), color) for i, color in enumerate(colors))



def make_colorscale(
    colorscale=None,
    colors=None,
    colormap=None,
    n_category=None,
    plot=True,
    layout_width=None,
    layout_height=None,
    title=None,
    html_file_path=None,
    plotly_html_file_path=None,
):

    if colorscale is not None:

        colorscale = colorscale

    elif colors is not None:

        colorscale = make_colorscale_from_colors(colors)

    elif colormap is not None:

        colorscale = make_colorscale_from_colors(get_colormap_colors(colormap))

    elif n_category is not None:

        colorscale = make_colorscale_from_colors(make_categorical_colors(n_category))

    if plot:

        x = tuple(range(len(colorscale)))

        colors = tuple(t[1] for t in colorscale)

        plot_and_save(
            dict(
                layout=dict(
                    width=layout_width,
                    height=layout_height,
                    title=title,
                    xaxis=dict(tickmode="array", tickvals=x, ticktext=colors),
                    yaxis=dict(ticks="", showticklabels=False),
                ),
                data=[
                    dict(
                        type="heatmap",
                        z=(x,),
                        colorscale=colorscale,
                        showscale=False,
                        hoverinfo="x+text",
                    )
                ],
            ),
            html_file_path,
            plotly_html_file_path,
        )

    return colorscale




def _normalize_nd_array(_nd_array, method, rank_method, raise_for_bad):

    is_good = ~check_nd_array_for_bad(_nd_array, raise_for_bad=raise_for_bad)

    nd_array_normalized = full(_nd_array.shape, nan)

    if is_good.any():

        nd_array_good = _nd_array[is_good]

        if method == "-0-":

            nd_array_good_std = nd_array_good.std()

            if nd_array_good_std == 0:

                nd_array_normalized[is_good] = 0

            else:

                nd_array_normalized[is_good] = (
                    nd_array_good - nd_array_good.mean()
                ) / nd_array_good_std

        elif method == "0-1":

            nd_array_good_min = nd_array_good.min()

            nd_array_good_range = nd_array_good.max() - nd_array_good_min

            if nd_array_good_range == 0:

                nd_array_normalized[is_good] = nan

            else:

                nd_array_normalized[is_good] = (
                    nd_array_good - nd_array_good_min
                ) / nd_array_good_range

        elif method == "sum":

            if nd_array_good.min() < 0:

                raise ValueError("Sum normalize only positives.")

            else:

                nd_array_good_sum = nd_array_good.sum()

                if nd_array_good_sum == 0:

                    nd_array_normalized[is_good] = 1 / is_good.sum()

                else:

                    nd_array_normalized[is_good] = nd_array_good / nd_array_good_sum

        elif method == "rank":

            nd_array_normalized[is_good] = rankdata(nd_array_good, method=rank_method)

    return nd_array_normalized


def normalize_nd_array(
    nd_array, axis, method, rank_method="average", raise_for_bad=True
):

    if axis is None:

        return _normalize_nd_array(nd_array, method, rank_method, raise_for_bad)

    else:

        return apply_along_axis(
            _normalize_nd_array, axis, nd_array, method, rank_method, raise_for_bad
        )


def _process_target_or_data_for_plotting(target_or_data, type, plot_std):

    if type == "continuous":

        if isinstance(target_or_data, Series):

            target_or_data = Series(
                normalize_nd_array(
                    target_or_data.values, None, "-0-", raise_for_bad=False
                ),
                name=target_or_data.name,
                index=target_or_data.index,
            )

        elif isinstance(target_or_data, DataFrame):

            target_or_data = DataFrame(
                normalize_nd_array(
                    target_or_data.values, 1, "-0-", raise_for_bad=False
                ),
                index=target_or_data.index,
                columns=target_or_data.columns,
            )

        target_or_data_nanmin = nanmin(target_or_data.values)

        target_or_data_nanmax = nanmax(target_or_data.values)

        if plot_std is None:

            plot_min = target_or_data_nanmin

            plot_max = target_or_data_nanmax

        else:

            plot_min = -plot_std

            plot_max = plot_std

        colorscale = make_colorscale(colormap="bwr", plot=False)

    else:

        plot_min = None

        plot_max = None

        if type == "categorical":

            n_color = unique(target_or_data).size

            colorscale = make_colorscale(colors=COLOR_CATEGORICAL[:n_color], plot=False)

        elif type == "binary":

            colorscale = make_colorscale(colors=COLOR_WHITE_BLACK, plot=False)

    return target_or_data, plot_min, plot_max, colorscale




def _ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays(
    _1d_array_0, _1d_array_1
):

    return apply_function_on_2_1d_arrays(
        _1d_array_0,
        _1d_array_1,
        lambda _1d_array_0, _1d_array_1: ((_1d_array_0 - _1d_array_1) ** 2).sum()
        ** (1 / 2),
        raise_for_bad=False,
    )




def get_1d_array_unique_objects_in_order(_1d_array, raise_for_bad=True):

    check_nd_array_for_bad(_1d_array, raise_for_bad=raise_for_bad)

    unique_objects_in_order = []

    for object_ in _1d_array:

        if object_ not in unique_objects_in_order:

            unique_objects_in_order.append(object_)

    return asarray(unique_objects_in_order)


def cluster_2d_array_slices(
    _2d_array,
    axis,
    groups=None,
    distance_function=None,
    linkage_method="average",
    optimal_ordering=True,
    raise_for_bad=True,
):

    #check_nd_array_for_bad(_2d_array, raise_for_bad=raise_for_bad)

    if axis == 1:

        _2d_array = _2d_array.T

    if distance_function is None:

        distance_function = (
            _ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays
        )

    if groups is None:

        return dendrogram(
            linkage(
                _2d_array,
                method=linkage_method,
                metric=distance_function,
                optimal_ordering=optimal_ordering,
            ),
            no_plot=True,
        )["leaves"]

    else:

        if len(groups) != _2d_array.shape[0]:

            raise ValueError(
                "len(groups) {} != len(axis-{} slices) {}".format(
                    len(groups), axis, _2d_array.shape[0]
                )
            )

        indices = []

        for i in get_1d_array_unique_objects_in_order(groups):

            group_indices = where(groups == i)[0]

            clustered_indices = dendrogram(
                linkage(
                    _2d_array[group_indices, :],
                    method=linkage_method,
                    metric=distance_function,
                    optimal_ordering=optimal_ordering,
                ),
                no_plot=True,
            )["leaves"]

            indices.append(group_indices[clustered_indices])

        return concatenate(indices)
        


    
def information_coefficient(x, y, n_grid=24, delta = 1.0):


    try:
        pearson_correlation = pearsonr(x, y)[0]
        
    except BaseException as err:
        print('Exception={} x={} y={}'.format(err, x, y))

    if isnan(pearson_correlation) or unique(x).size == 1 or unique(y).size == 1:

        return nan

    else:

        pearson_correlation_abs = abs(pearson_correlation)

       # xr = pandas2ri.py2ri(x)
       # yr = pandas2ri.py2ri(y)

        #print('IC: sum x:{} sum y:{}'.format(np.sum(x), np.sum(y)))
        
        #print('bandwidth x: {}'.format(mass.bcv(x)[0]))
        bandwidth_x = delta * mass.bcv(x)[0] * (1 - pearson_correlation_abs * 0.75)

        #print('bandwidth y: {}'.format(mass.bcv(y)[0]))
        bandwidth_y = delta * mass.bcv(y)[0] * (1 - pearson_correlation_abs * 0.75)

        #print('bandwidth... done')
        
        fxy = (
            asarray(
                mass.kde2d(
                    x, y, asarray((bandwidth_x, bandwidth_y)), n=asarray((n_grid,))
                )[2]
            )
            + eps
        )

        #dx = (x.max() - x.min()) / (n_grid - 1)

        #dy = (y.max() - y.min()) / (n_grid - 1)

        pxy = fxy / (fxy.sum())

        px = pxy.sum(axis=1)
        px = px/px.sum()

        py = pxy.sum(axis=0)
        py = py/py.sum()

        #mi = (pxy * log2(pxy / (asarray((px,)).T * asarray((py,))))).sum()

        hxy = - (pxy * log2(pxy)).sum() 
        hx = -(px * log2(px)).sum() 
        hy = -(py * log2(py)).sum() 
        mi = hx + hy - hxy

        # The mutual information is normalized as in  Linfoot's Information Coefficient (https://www.sciencedirect.com/science/article/pii/S001999585790116X)
        # using the mutual information for a Gaussian distribution (see e.g. Example 8.5.1 in Elements of Information Theory 2nd ed - T. Cover, J. Thomas Wiley, 2006)

        IC = sign(pearson_correlation) * sqrt(1 - power(2.0, -2.0 * mi))

        return IC
        


def nd_array_is_sorted(nd_array, raise_for_bad=True):

    check_nd_array_for_bad(nd_array, raise_for_bad=raise_for_bad)

    diff_ = diff(nd_array)

    return (diff_ <= 0).all() or (0 <= diff_).all()


def plot_and_save(figure, html_file_path, plotly_html_file_path):

    if html_file_path is not None:

        print(
            offline_plot(
                figure, filename=html_file_path, auto_open=False, show_link=False
            )
        )

    if plotly_html_file_path is not None:

        print(
            plotly_plot(
                figure,
                filename=plotly_html_file_path,
                file_opt="overwrite",
                sharing="public",
                auto_open=False,
                show_link=False,
            )
        )

    iplot(figure, show_link=False)
    


def match_features_against_phenotype(
    target,
    data,
    target_ascending=True,
    score_moe_p_value_fdr=None,
    n_job=1,
    match_function=information_coefficient,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    n_extreme=8,
    fraction_extreme=None,
    random_seed=20_121_020,
    n_sampling=0,
    n_permutation=0,
    score_ascending=False,
    plot_only_sign=None,
    target_type="continuous",
    cluster_within_category=True,
    data_type="continuous",
    plot_std=None,
    title=None,
    layout_width=880,
    row_height=64,
    layout_side_margin=196,
    annotation_font_size=10,
    file_path_prefix=None,
    plotly_html_file_path_prefix=None):

    common_indices = target.index & data.columns

    print(
        "target.index ({}) & data.columns ({}) have {} in common.".format(
            target.index.size, data.columns.size, len(common_indices)
        )
    )

    target = target[common_indices]

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    data = data[target.index]

    if score_moe_p_value_fdr is None:

        score_moe_p_value_fdr = _match(
            target.values,
            data.values,
            n_job,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
            n_extreme,
            fraction_extreme,
            random_seed,
            n_sampling,
            n_permutation,
        )

        if score_moe_p_value_fdr.isna().values.all():

            return score_moe_p_value_fdr

        score_moe_p_value_fdr.index = data.index

        score_moe_p_value_fdr.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        if file_path_prefix is not None:

            score_moe_p_value_fdr.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    else:

        score_moe_p_value_fdr = score_moe_p_value_fdr.reindex(index=data.index)

    scores_to_plot = score_moe_p_value_fdr.copy()

    if n_extreme is not None or fraction_extreme is not None:

        scores_to_plot = score_moe_p_value_fdr.loc[
            select_series_indices(
                score_moe_p_value_fdr["Score"],
                "<>",
                n=n_extreme,
                fraction=fraction_extreme,
                plot=False,
            )
        ]

    if plot_only_sign is not None:

        if plot_only_sign == "-":

            indices = scores_to_plot["Score"] <= 0

        elif plot_only_sign == "+":

            indices = 0 <= scores_to_plot["Score"]

        scores_to_plot = scores_to_plot.loc[indices]

    scores_to_plot.sort_values("Score", ascending=score_ascending, inplace=True)

    data_to_plot = data.loc[scores_to_plot.index]

    annotations = _make_annotations(scores_to_plot)

    target, target_plot_min, target_plot_max, target_colorscale = _process_target_or_data_for_plotting(
        target, target_type, plot_std
    )

    if (
        cluster_within_category
        and target_type in ("binary", "categorical")
        and 1 < target.value_counts().min()
        and nd_array_is_sorted(target.values)
        and not data_to_plot.isna().all().any()
    ):

        print("Clustering heat map within category ...")

        clustered_indices = cluster_2d_array_slices(
            data_to_plot.values, 1, groups=target.values, raise_for_bad=False
        )

        target = target.iloc[clustered_indices]

        data_to_plot = data_to_plot.iloc[:, clustered_indices]

    data_to_plot, data_plot_min, data_plot_max, data_colorscale = _process_target_or_data_for_plotting(
        data_to_plot, data_type, plot_std
    )

    target_row_fraction = max(0.01, 1 / (data_to_plot.shape[0] + 2))

    target_yaxis_domain = (1 - target_row_fraction, 1)

    data_yaxis_domain = (0, 1 - target_row_fraction * 2)

    data_row_fraction = (
        data_yaxis_domain[1] - data_yaxis_domain[0]
    ) / data_to_plot.shape[0]

    layout = dict(
        width=layout_width,
        height=row_height * max(8, (data_to_plot.shape[0] + 2) ** 0.8),
        margin=dict(l=layout_side_margin, r=layout_side_margin),
        xaxis=dict(anchor="y", tickfont=dict(size=annotation_font_size)),
        yaxis=dict(
            domain=data_yaxis_domain, dtick=1, tickfont=dict(size=annotation_font_size)
        ),
        yaxis2=dict(
            domain=target_yaxis_domain, tickfont=dict(size=annotation_font_size)
        ),
        title=title,
        annotations=[],
    )

    data = [
        dict(
            yaxis="y2",
            type="heatmap",
            z=target.to_frame().T.values,
            x=target.index,
            y=(target.name,),
            text=(target.index,),
            zmin=target_plot_min,
            zmax=target_plot_max,
            colorscale=target_colorscale,
            showscale=False,
        ),
        dict(
            yaxis="y",
            type="heatmap",
            z=data_to_plot.values[::-1],
            x=data_to_plot.columns,
            y=data_to_plot.index[::-1],
            zmin=data_plot_min,
            zmax=data_plot_max,
            colorscale=data_colorscale,
            showscale=False,
        ),
    ]

    layout_annotation_template = dict(
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="middle",
        font=dict(size=annotation_font_size),
        width=64,
        showarrow=False,
    )

    for annotation_index, (annotation, strs) in enumerate(annotations.items()):

        x = 1.0016 + annotation_index / 10

        layout["annotations"].append(
            dict(
                x=x,
                y=target_yaxis_domain[1] - (target_row_fraction / 2),
                text="<b>{}</b>".format(annotation),
                **layout_annotation_template,
            )
        )

        y = data_yaxis_domain[1] - (data_row_fraction / 2)

        for str_ in strs:

            layout["annotations"].append(
                dict(
                    x=x,
                    y=y,
                    text="<b>{}</b>".format(str_),
                    **layout_annotation_template,
                )
            )

            y -= data_row_fraction

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    if plotly_html_file_path_prefix is None:

        plotly_html_file_path = None

    else:

        plotly_html_file_path = "{}.html".format(plotly_html_file_path_prefix)

    plot_and_save(
        dict(layout=layout, data=data),
        html_file_path=html_file_path,
        plotly_html_file_path=plotly_html_file_path,
    )

    return score_moe_p_value_fdr

def read_gmts(gmt_file_paths, sets=None, drop_description=True, collapse=False):

    dfs = []

    for gmt_file_path in gmt_file_paths:

        dfs.append(read_gmt(gmt_file_path, drop_description=drop_description))

    df = concat(dfs, sort=True)

    if sets is not None:

        df = df.loc[(df.index & sets)].dropna(axis=1, how="all")

    if collapse:

        return df.unstack().dropna().sort_values().unique()

    else:

        return df
        

def read_gmt(gmt_file_path, drop_description=True):

    lines = []

    with open(gmt_file_path) as gmt_file:

        for line in gmt_file:

            split = line.strip().split(sep="\t")

            lines.append(split[:2] + [gene for gene in set(split[2:]) if gene])

    df = DataFrame(lines)

    df.set_index(0, inplace=True)

    df.index.name = "Gene Set"

    if drop_description:

        df.drop(1, axis=1, inplace=True)

        df.columns = tuple("Gene {}".format(i) for i in range(0, df.shape[1]))

    else:

        df.columns = ("Description",) + tuple(
            "Gene {}".format(i) for i in range(0, df.shape[1] - 1)
        )

    return df

def multiprocess(callable_, args, n_job, random_seed=20121020):

    seed(random_seed)

    with Pool(n_job) as process:

        return process.starmap(callable_, args)

def split_df(df, axis, n_split):

    if not (0 < n_split <= df.shape[axis]):

        raise ValueError(
            "Invalid: 0 < n_split ({}) <= n_slices ({})".format(n_split, df.shape[axis])
        )

    n = df.shape[axis] // n_split

    dfs = []

    for i in range(n_split):

        start_i = i * n

        end_i = (i + 1) * n

        if axis == 0:

            dfs.append(df.iloc[start_i:end_i])

        elif axis == 1:

            dfs.append(df.iloc[:, start_i:end_i])

    i = n * n_split

    if i < df.shape[axis]:

        if axis == 0:

            dfs.append(df.iloc[i:])

        elif axis == 1:

            dfs.append(df.iloc[:, i:])

    return dfs


def single_sample_gseas(
    gene_x_sample,
    gene_sets,
    statistic="ks",
    alpha=1.0,
    n_job=1,
    file_path=None,
        sample_norm_type = None):

    score__gene_set_x_sample = concat(
        multiprocess(
            _single_sample_gseas,
            (
                (gene_x_sample, gene_sets_, statistic, alpha, sample_norm_type)
                for gene_sets_ in split_df(gene_sets, 0, min(gene_sets.shape[0], n_job))
            ),
            n_job,
        )
    )

    if file_path is not None:

        score__gene_set_x_sample.to_csv(file_path, sep="\t")

    return score__gene_set_x_sample




def _single_sample_gseas(gene_x_sample,
                         gene_sets,
                         statistic,
                         alpha,
                         sample_norm_type):

    print("Running single-sample GSEA with {} gene sets ...".format(gene_sets.shape[0]))

    score__gene_set_x_sample = full((gene_sets.shape[0], gene_x_sample.shape[1]), nan)

    for sample_index, (sample_name, gene_score) in enumerate(gene_x_sample.items()):

        for gene_set_index, (gene_set_name, gene_set_genes) in enumerate(
            gene_sets.iterrows()):

            score__gene_set_x_sample[gene_set_index, sample_index] = single_sample_gsea(
                                                                                      gene_score,
                                                                                      gene_set_genes,
                                                                                      statistic=statistic,
                                                                                      alpha=alpha,
                                                                                      plot=False,
                                                                                      sample_norm_type = sample_norm_type)

    score__gene_set_x_sample = DataFrame(
        score__gene_set_x_sample, index=gene_sets.index, columns=gene_x_sample.columns
    )

    return score__gene_set_x_sample



def single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    alpha=1.0,
    plot=False,
    plot_gene_names = False,
    title=None,
    gene_score_name=None,
    annotation_text_font_size=12,
    annotation_text_width=100,
    annotation_text_yshift=50,
    html_file_path=None,
    sample_norm_type = None,
    plotly_html_file_path=None,
):

    if sample_norm_type == 'rank':
        gene_score = gene_score.rank(method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)
        gene_score = 10000 * (gene_score - gene_score.min())/(gene_score.max() - gene_score.min())
    elif sample_norm_type == 'zscore':
        gene_score = (gene_score - gene_score.mean())/gene_score.std()
    elif sample_norm_type is not None:
        sys.exit('ERROR: unknown sample_norm_type: {}'.format(sample_norm_type))
        
    gene_score_sorted = gene_score.sort_values(ascending=False)
    
    gene_set_gene_None = {gene_set_gene: None for gene_set_gene in gene_set_genes}

    in_ = asarray(
        [
            gene_score_gene in gene_set_gene_None
            #for gene_score_gene in gene_score.index.values
            for gene_score_gene in gene_score_sorted.index.values
        ],
        dtype=int,
    )

    #print(in_)
    #print(gene_score_sorted)
    
    up = in_ * absolute(gene_score_sorted.values)**alpha
    up /= up.sum()
    down = 1.0 - in_
    down /= down.sum()
    cumsum = (up - down).cumsum()
    up_CDF = up.cumsum()
    down_CDF = down.cumsum()

    if statistic == "ks":

        max_ = cumsum.max()
        min_ = cumsum.min()
        if absolute(min_) < absolute(max_):
            gsea_score = max_
        else:
            gsea_score = min_
            
    elif statistic == "auc":
        gsea_score = cumsum.sum()

    gsea_score = round(gsea_score, 3)
        
    '''if plot:
        
        _plot_mountain(
            up_CDF,
            down_CDF,
            cumsum,
            in_,
            gene_score_sorted,
            gsea_score,
            None,
            None,
            title,
            gene_score_name,
            annotation_text_font_size,
            annotation_text_width,
            annotation_text_yshift,
            plot_gene_names,
            html_file_path,
            plotly_html_file_path,
        )'''

    return gsea_score
