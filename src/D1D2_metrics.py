import numpy as np


class D1D2:
    def get_window(tg, W):
        """
        Docstring for get_window

        :param tg: int: true drift point
        :param W: int: Constant offset
        """
        W = round(W)
        if W <= 0:
            raise ValueError("Window size must be a value above 0")
        return tg - W, tg + W

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
                    # break  # one hit per drift

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

