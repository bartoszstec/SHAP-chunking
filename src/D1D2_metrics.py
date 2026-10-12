import numpy as np
import math

class D1D2:
    def get_window(tg, W, detection_delay_tolerance=500):
        """
        Docstring for get_window

        :param tg: int: true drift point
        :param W: int: Constant offset
        """
        W = round(W)
        if W <= 0:
            raise ValueError("Window size must be a value above 0")
        #return tg - W, tg + W
        return tg - W/ 2, tg + W / 2 + detection_delay_tolerance



    def tpr(true_drifts: list[int], detected_points: list[int], W: int) -> float | None:
        """
        Docstring for TPR
        True positive rate - the ratio of true drifts accruately detected to all true drifts.

        .. math::
            TPR = \\frac{|t_g \\in T_g : \\exists t_d \\in T_d \\land t_d \\in DW(t_g)|}{|T_g|}

        :param true_drifts: list of true drift points
        :param detected: list of detected drift points
        :param W: detection window size., must be a positive integer. A detection window in the form of [x-W,x+W] is created around each true drift point.
        """

        matched = set()
        used_detections = set()

        for tg in true_drifts:
            window_start, window_end = D1D2.get_window(tg, W)
            for i, tf in enumerate(detected_points):
                if i not in used_detections and window_start <= tf <= window_end:
                    matched.add(tg)
                    used_detections.add(i)
                    break  # only one detection per drift

        tpr = len(matched) / len(true_drifts) if true_drifts else 0.0
        return tpr

    def fdr(true_drifts: list[int], detected_points: list[int], W: int) -> float | None:
        """
        Docstring for FDR
        False discovery rate - the ratio of the drifts detected outside the detection window to all detected drifts.

        .. math::
            FDR = \\frac{|t_d \\in T_d : t_d \\notin EDW|}{|T_d|}

        :param true_drifts: list of true drift points
        :param detected: list of detected drift points
        :param W: detection window size., must be a positive integer. A detection window in the form of [x-W,x+W] is created around each true drift point.
        """
        if not detected_points:
            return None

        matched_detections = set()

        for tg in true_drifts:
            window_start, window_end = D1D2.get_window(tg, W)
            for i, tf in enumerate(detected_points):
                if i not in matched_detections and window_start <= tf <= window_end:
                    matched_detections.add(i)
                    break  # one hit per drift

        num_fp = len(detected_points) - len(matched_detections)
        fdr = num_fp / len(detected_points) #if detected_points else 0.0
        return fdr

    def fedp(true_drifts: list[int], detected_points: list[int], W: int) -> list[tuple[int, int, int]]:
        """
        FEDP = tf - tg, gdzie tf to PIERWSZY (w sensie czasowym) punkt detekcji w oknie EDR.
        Jeśli brak tf w oknie, dany tg nie wnosi do średniej.
        Uwaga: NIE sortujemy detected_points — zachowujemy chronologię wejścia.
        """
        fedp = []
        for tg in true_drifts:
            window_start, window_end = D1D2.get_window(tg, W)
            # Pierwszy skuteczny punkt w KOLEJNOŚCI 'detected_points'
            first_tf = next(
                (tf for tf in detected_points if window_start <= tf <= window_end), None
            )
            if first_tf is not None:
                fedp.append((tg, first_tf, first_tf - tg))
        return fedp

    def compute_eddr(true_drifts, detected_points, W, max_effective=3):
        effective_detections = 0
        for tg in true_drifts:
            window_start, window_end = D1D2.get_window(tg, W)
            if any(window_start <= tf <= window_end for tf in detected_points):
                effective_detections += 1
        return 1 if 1 <= effective_detections <= max_effective else 0

    def D1(true_drifts: list[int], detected_points: list[int]) -> float | None:
        """
         Docstring for D1
         D1 - the average distance from each detected drift point to the closest true drift point

         .. math::
            D_1 = \\frac{1}{|T_d|}\\sum_{t_d \\in T_d}\\min_{t_g \\in T_g}|t_d - t_g|

        :param true_drifts: list of true drift points
         :param detected: list of detected drift points
        """
        if not detected_points or not true_drifts:
            return None
        d1 = np.mean(
            [min(abs(tf - tg) for tg in true_drifts) for tf in detected_points]
        )
        return float(d1)

    def D2(true_drifts: list[int], detected_points: list[int]) -> float | None:
        """
        Docstring for D2
        D2 - the average distance from each true drift point to the closest detected drift point

        .. math::
            D_2 = \\frac{1}{|T_g|}\\sum_{t_g \\in T_g}\\min_{t_d \\in T_d}|t_g - t_d|

        :param true_drifts: list of true drift points
        :param detected: list of detected drift points
        """
        if not detected_points or not true_drifts:
            return None
        d2 = np.mean(
            [min(abs(tg - tf) for tf in detected_points) for tg in true_drifts]
        )
        return float(d2)

    def b_mean(true_drifts, detected_points, W):
        tpr = D1D2.tpr(true_drifts, detected_points, W)
        fdr = D1D2.fdr(true_drifts, detected_points, W)

        return (tpr + (1 - fdr)) / 2

    def g_mean(true_drifts, detected_points, W):
        tpr = D1D2.tpr(true_drifts, detected_points, W)
        fdr = D1D2.fdr(true_drifts, detected_points, W)

        return math.sqrt(tpr * (1 - fdr))


class StatisticalComparison:
    """Utilities for statistically comparing multiple algorithms.

    The main use case is a comparison of several algorithms evaluated on the
    same collection of datasets, data streams, or experimental scenarios.
    """

    @staticmethod
    def critical_difference_diagram(
        results,
        lower_is_better=True,
        alpha=0.05,
        title="Critical Difference Diagram",
        save_path=None,
        show=True,
        reverse_x=True,
        show_rank_values=True,
        rank_decimals=4,
        figsize=None,
        colors=None,
    ):
        """Run Friedman + Nemenyi analysis and draw a classic Critical Difference Diagram.

        The visual style follows the commonly used CD-diagram layout:
        the rank axis is placed at the top, algorithms are listed on the left
        and right side, and horizontal thick bars denote groups of methods that
        do not differ significantly according to the Nemenyi post-hoc test.

        This implementation does not require ``scikit-posthocs`` or ``networkx``.
        It uses only common scientific Python packages: pandas, matplotlib,
        numpy, and scipy.

        Parameters
        ----------
        results : pandas.DataFrame
            Rows correspond to datasets / streams / scenarios.
            Columns correspond to algorithms. Values are metric scores.
            For D1 and D2 metrics, ``lower_is_better=True`` is appropriate.
        lower_is_better : bool, default=True
            Whether lower metric values indicate better performance.
        alpha : float, default=0.05
            Significance level used for Friedman and Nemenyi reporting.
        title : str, default="Critical Difference Diagram"
            Figure title. Pass ``None`` or an empty string to hide it.
        save_path : str or pathlib.Path, optional
            If provided, the figure is saved to this location.
        show : bool, default=True
            Whether to call ``plt.show()``.
        reverse_x : bool, default=True
            If True, the rank scale is displayed as in many published CD diagrams:
            larger ranks on the left and rank 1 on the right.
        show_rank_values : bool, default=True
            Whether to print average rank values next to algorithm names.
        rank_decimals : int, default=4
            Number of decimal places used when printing average ranks.
        figsize : tuple, optional
            Matplotlib figure size. If omitted, the height is adjusted to the
            number of algorithms.
        colors : dict or list, optional
            Colors of the algorithms (``{name: color}`` or a list in column
            order). By default a colorblind-safe palette is assigned in column
            order, so an algorithm keeps the same color in every diagram
            regardless of its rank.

        Returns
        -------
        dict
            A dictionary with:
            - ``friedman_statistic``
            - ``friedman_pvalue``
            - ``average_ranks``
            - ``nemenyi_pvalues``
            - ``significance_matrix``
            - ``critical_difference``
            - ``non_significant_groups``
            - ``figure``
            - ``axis``
        """
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from scipy.stats import friedmanchisquare, studentized_range
        except ImportError as exc:
            raise ImportError(
                "critical_difference_diagram requires pandas, matplotlib, numpy, "
                "and scipy."
            ) from exc

        if not isinstance(results, pd.DataFrame):
            results = pd.DataFrame(results)

        if results.shape[1] < 3:
            raise ValueError(
                "Critical Difference Diagram requires results for at least 3 algorithms."
            )
        if results.shape[0] < 2:
            raise ValueError(
                "At least 2 datasets / streams / scenarios are required for comparison."
            )
        if results.isnull().any().any():
            raise ValueError("The results table cannot contain missing values.")

        # Ranks: 1 = best algorithm in a given row/scenario.
        ranks = results.rank(axis=1, ascending=lower_is_better, method="average")
        average_ranks = ranks.mean(axis=0).sort_values()

        # Friedman test on repeated measurements across algorithms.
        friedman_input = [results[col].to_numpy() for col in results.columns]
        friedman_statistic, friedman_pvalue = friedmanchisquare(*friedman_input)

        # Nemenyi post-hoc comparison computed directly from the Studentized Range distribution.
        n_blocks = results.shape[0]
        n_algorithms = results.shape[1]
        standard_error = math.sqrt(n_algorithms * (n_algorithms + 1) / (6.0 * n_blocks))

        # In the Nemenyi test, q_alpha is the Studentized Range critical value divided by sqrt(2).
        q_alpha = studentized_range.ppf(1.0 - alpha, n_algorithms, np.inf) / math.sqrt(2.0)
        critical_difference = float(q_alpha * standard_error)

        algs = list(results.columns)
        nemenyi_pvalues = pd.DataFrame(
            np.ones((n_algorithms, n_algorithms), dtype=float),
            index=algs,
            columns=algs,
        )

        for i, alg_i in enumerate(algs):
            for j in range(i + 1, n_algorithms):
                alg_j = algs[j]
                avg_rank_diff = abs(float(average_ranks[alg_i]) - float(average_ranks[alg_j]))
                q_stat = (avg_rank_diff / standard_error) * math.sqrt(2.0)
                p_value = float(studentized_range.sf(q_stat, n_algorithms, np.inf))
                nemenyi_pvalues.loc[alg_i, alg_j] = p_value
                nemenyi_pvalues.loc[alg_j, alg_i] = p_value

        # True means: no statistically significant difference for the selected alpha.
        significance_matrix = nemenyi_pvalues >= alpha

        sorted_ranks = average_ranks.sort_values()
        sorted_algs = list(sorted_ranks.index)
        sorted_values = np.array([float(sorted_ranks[a]) for a in sorted_algs])

        # -------------------------------------------------
        # Maximal contiguous groups without significant differences.
        # -------------------------------------------------
        def interval_is_non_significant(i, j):
            for left in range(i, j + 1):
                for right in range(left + 1, j + 1):
                    alg_left = sorted_algs[left]
                    alg_right = sorted_algs[right]
                    if not bool(significance_matrix.loc[alg_left, alg_right]):
                        return False
            return True

        intervals = []
        if float(friedman_pvalue) >= alpha:
            intervals = [(0, n_algorithms - 1)]
        else:
            for i in range(n_algorithms):
                for j in range(i + 1, n_algorithms):
                    if interval_is_non_significant(i, j):
                        intervals.append((i, j))

        maximal_intervals = []
        for candidate in intervals:
            is_contained = any(
                other != candidate
                and other[0] <= candidate[0]
                and other[1] >= candidate[1]
                for other in intervals
            )
            if not is_contained:
                maximal_intervals.append(candidate)

        # Remove duplicates while preserving order.
        seen_intervals = set()
        maximal_intervals = [
            interval for interval in maximal_intervals
            if not (interval in seen_intervals or seen_intervals.add(interval))
        ]

        non_significant_groups = [
            [sorted_algs[pos] for pos in range(i, j + 1)]
            for i, j in maximal_intervals
        ]

        # -------------------------------------------------
        # Publication-like CD diagram inspired by the supplied example.
        # -------------------------------------------------
        # Colorblind-safe categorical palette; the color follows the algorithm, not its rank.
        default_palette = [
            "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#7d55c7", "#8a6d3b",
        ]
        if colors is None:
            color_of = {alg: default_palette[i % len(default_palette)] for i, alg in enumerate(algs)}
        elif isinstance(colors, dict):
            color_of = {alg: colors.get(alg, "0.45") for alg in algs}
        else:
            color_of = {alg: colors[i % len(colors)] for i, alg in enumerate(algs)}

        ink = "#262626"
        group_color = "#4a4a4a"

        if figsize is None:
            figsize = (10.5, max(3.8, 0.34 * n_algorithms + 1.8))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")

        if title:
            ax.set_title(title, pad=18, fontsize=12, color=ink)

        k = n_algorithms
        x_pad = max(0.55, k / 14.0)
        min_rank = 1.0
        max_rank = float(k)
        x_left_label = max_rank + x_pad
        x_right_label = min_rank - x_pad

        if reverse_x:
            ax.set_xlim(max_rank + 1.8 * x_pad, min_rank - 1.8 * x_pad)
        else:
            ax.set_xlim(min_rank - 1.8 * x_pad, max_rank + 1.8 * x_pad)

        # Split methods into two sides, as in classic CD diagrams.
        changepoint = int(math.floor(k / 2.0))
        right_side_indices = list(range(changepoint))  # best ranks
        left_side_indices = list(range(changepoint, k))  # worse ranks

        # Rank axis at the top.
        axis_y = 0.0
        ax.hlines(axis_y, min_rank, max_rank, linewidth=1.2, color=ink)
        for tick in range(1, k + 1):
            ax.vlines(tick, axis_y - 0.08, axis_y + 0.08, linewidth=1.0, color=ink)
            ax.text(tick, axis_y + 0.16, str(tick), ha="center", va="bottom", fontsize=10, color=ink)

        # Minor guide marks for a slightly more technical appearance.
        if k <= 6:
            minor_steps = [0.25, 0.50, 0.75]
        elif k <= 10:
            minor_steps = [0.50]
        else:
            minor_steps = []
        for base in range(1, k):
            for step in minor_steps:
                ax.vlines(base + step, axis_y - 0.045, axis_y + 0.045, linewidth=0.6, color=ink)

        # Horizontal group bars just below the axis.
        group_base_y = -0.28
        group_step = 0.18
        for level, (i, j) in enumerate(maximal_intervals):
            y = group_base_y - level * group_step
            ax.plot(
                [sorted_values[i], sorted_values[j]], [y, y],
                linewidth=4.0, color=group_color, solid_capstyle="round", zorder=3,
            )

        # Treatment connectors and labels.
        elbow_drop = 0.10
        row_step = 0.34
        label_gap = 0.08

        # Determine vertical rows. The order on each side follows the ranking order.
        rows_for_right = {idx: pos + 1 for pos, idx in enumerate(right_side_indices)}
        rows_for_left = {idx: pos + 1 for pos, idx in enumerate(reversed(left_side_indices))}

        def label_text(alg, rank_value, side):
            if not show_rank_values:
                return str(alg)
            rank_str = f"{rank_value:.{rank_decimals}f}"
            if side == "left":
                return f"{alg}   {rank_str}"
            return f"{rank_str}   {alg}"

        max_row = 1
        for idx, (alg, rank_value) in enumerate(zip(sorted_algs, sorted_values)):
            if idx in right_side_indices:
                side = "right"
                label_x = x_right_label
                row = rows_for_right[idx]
                ha = "left" if reverse_x else "right"
                text_x = label_x - label_gap if reverse_x else label_x + label_gap
            else:
                side = "left"
                label_x = x_left_label
                row = rows_for_left[idx]
                ha = "right" if reverse_x else "left"
                text_x = label_x + label_gap if reverse_x else label_x - label_gap

            max_row = max(max_row, row)
            label_y = -0.78 - (row - 1) * row_step
            elbow_y = label_y + elbow_drop
            alg_color = color_of[alg]
            ax.plot(
                [rank_value, rank_value, label_x],
                [axis_y, elbow_y, elbow_y],
                linewidth=1.6,
                color=alg_color,
                solid_joinstyle="round",
                zorder=2,
            )
            # Rank position on the axis, with a white ring separating it from the axis line.
            ax.plot(
                rank_value, axis_y, marker="o", markersize=6,
                markerfacecolor=alg_color, markeredgecolor="white", markeredgewidth=1.0,
                zorder=4,
            )
            ax.text(
                text_x,
                elbow_y,
                label_text(alg, rank_value, side),
                ha=ha,
                va="center",
                fontsize=10,
                color=ink,
            )

        # CD reference segment in the upper-left portion of the plot.
        cd_y = 0.58
        cd_x_start = max_rank if reverse_x else min_rank
        cd_x_end = max(min_rank, max_rank - critical_difference) if reverse_x else min(max_rank, min_rank + critical_difference)
        ax.hlines(cd_y, cd_x_start, cd_x_end, linewidth=2.2, color=ink)
        ax.vlines([cd_x_start, cd_x_end], cd_y - 0.05, cd_y + 0.05, linewidth=1.2, color=ink)
        ax.text(
            (cd_x_start + cd_x_end) / 2.0,
            cd_y + 0.10,
            f"CD = {critical_difference:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=ink,
        )

        # Clean frame and set y-range.
        lowest_group_y = group_base_y - max(0, len(maximal_intervals) - 1) * group_step
        lowest_label_y = -0.78 - max(0, max_row - 1) * row_step
        y_min = min(lowest_group_y - 0.18, lowest_label_y - 0.20)
        ax.set_ylim(y_min, cd_y + 0.42)
        ax.axis("off")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()

        return {
            "friedman_statistic": float(friedman_statistic),
            "friedman_pvalue": float(friedman_pvalue),
            "average_ranks": average_ranks,
            "nemenyi_pvalues": nemenyi_pvalues,
            "significance_matrix": significance_matrix,
            "critical_difference": critical_difference,
            "non_significant_groups": non_significant_groups,
            "figure": fig,
            "axis": ax,
        }
