class _ODEFitResult:
    order: int
    params: np.ndarray
    bic: float
    rss: float
    n_obs: int
    cond_number: float
    is_stable: bool
    feature: np.ndarray
    roots: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))


class _ODERegimeKernel:
    """
    ODE-based regime discovery:
    fit local 0/1/2-order models on sliding windows, choose the best order,
    then cluster valid windows within each order using DBSCAN.
    """

    def __init__(self, cfg: ConformalPredictionConfig):
        self.cfg = cfg
        self.max_regimes = int(cfg.max_regimes)
        self.refit_every = max(1, int(getattr(cfg, "ode_refit_every", 20)))
        self.bootstrap_size = max(1, int(getattr(cfg, "ode_bootstrap_size", 40)))
        self.assign_threshold = float(getattr(cfg, "ode_assignment_threshold", 2.5))
        self.cluster_min_samples = max(2, int(getattr(cfg, "ode_cluster_min_samples", 5)))
        self.cond_max = float(getattr(cfg, "ode_cond_max", 1e8))
        self.stable_only = bool(getattr(cfg, "ode_stable_only", True))
        self.min_samples = max(8, int(getattr(cfg, "ode_min_samples", 12)))
        self.order_switch_margin = float(getattr(cfg, "ode_order_switch_margin", 0.0))
        self.order_switch_patience = max(1, int(getattr(cfg, "ode_order_switch_patience", 1)))
        self.use_feature_filter = bool(getattr(cfg, "ode_use_feature_filter", False))
        self.filter_process_var = max(1e-8, float(getattr(cfg, "ode_filter_process_var", 0.05)))
        self.filter_measure_var = max(1e-8, float(getattr(cfg, "ode_filter_measure_var", 0.5)))
        self.filter_init_var = max(1e-8, float(getattr(cfg, "ode_filter_init_var", 1.0)))
        self.filter_reset_on_order_change = bool(getattr(cfg, "ode_filter_reset_on_order_change", True))

        self.window_history: List[_ODEFitResult] = []
        self.cluster_centers: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self.cluster_rids: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        self.next_rid: int = 0
        self.prev_rid: Optional[int] = None
        self.prev_order: Optional[int] = None
        self.pending_order: Optional[int] = None
        self.pending_order_hits: int = 0
        self._steps_seen: int = 0
        self._filter_mean: Dict[int, Optional[np.ndarray]] = {0: None, 1: None, 2: None}
        self._filter_cov: Dict[int, Optional[np.ndarray]] = {0: None, 1: None, 2: None}

    def reset(self) -> None:
        self.window_history.clear()
        self.cluster_centers = {0: [], 1: [], 2: []}
        self.cluster_rids = {0: [], 1: [], 2: []}
        self.next_rid = 0
        self.prev_rid = None
        self.prev_order = None
        self.pending_order = None
        self.pending_order_hits = 0
        self._steps_seen = 0
        self._filter_mean = {0: None, 1: None, 2: None}
        self._filter_cov = {0: None, 1: None, 2: None}

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        w = max(1, int(getattr(self.cfg, "ode_smooth_window", 1)))
        if w <= 1 or len(x) < w:
            return x
        ker = np.ones(w, dtype=float) / float(w)
        return np.convolve(x, ker, mode="same")

    def _bic(self, rss: float, n_obs: int, n_params: int) -> float:
        rss_eff = max(float(rss), 1e-12)
        return float(n_obs * np.log(rss_eff / max(n_obs, 1)) + n_params * np.log(max(n_obs, 1)))

    def _solve_lstsq(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = float(np.dot(resid, resid))
        cond = float(np.linalg.cond(X)) if X.size > 0 else np.inf
        return beta, rss, cond

    def _fit_order0(self, x: np.ndarray) -> _ODEFitResult:
        b0 = float(np.mean(x))
        resid = x - b0
        rss = float(np.dot(resid, resid))
        feature = np.array([b0], dtype=float)
        return _ODEFitResult(
            order=0,
            params=np.array([b0], dtype=float),
            bic=self._bic(rss, len(x), 1),
            rss=rss,
            n_obs=len(x),
            cond_number=1.0,
            is_stable=True,
            feature=feature,
        )

    def _fit_order1(self, x: np.ndarray) -> Optional[_ODEFitResult]:
        if len(x) < self.min_samples:
            return None
        dx = np.diff(x)
        x_prev = x[:-1]
        if len(dx) < self.min_samples - 1:
            return None
        X = np.column_stack([-x_prev, np.ones_like(x_prev)])
        beta, rss, cond = self._solve_lstsq(X, dx)
        a0, b0 = float(beta[0]), float(beta[1])
        is_stable = bool(a0 >= 0.0)
        feature = np.array([a0, b0], dtype=float)
        return _ODEFitResult(
            order=1,
            params=np.array([a0, b0], dtype=float),
            bic=self._bic(rss, len(dx), 2),
            rss=rss,
            n_obs=len(dx),
            cond_number=cond,
            is_stable=is_stable,
            feature=feature,
        )

    def _fit_order2(self, x: np.ndarray) -> Optional[_ODEFitResult]:
        if len(x) < self.min_samples + 1:
            return None
        dx = np.diff(x)
        d2x = np.diff(dx)
        if len(d2x) < self.min_samples - 2:
            return None
        dx_prev = dx[:-1]
        x_prev = x[:-2]
        X = np.column_stack([-dx_prev, -x_prev, np.ones_like(x_prev)])
        beta, rss, cond = self._solve_lstsq(X, d2x)
        a1, a0, b0 = float(beta[0]), float(beta[1]), float(beta[2])
        roots = np.roots(np.array([1.0, a1, a0], dtype=float))
        is_stable = bool(np.all(np.real(roots) <= 1e-8))
        if np.iscomplexobj(roots):
            roots_sorted = sorted(roots, key=lambda z: (np.real(z), np.imag(z)))
        else:
            roots_sorted = sorted([complex(r) for r in roots], key=lambda z: (np.real(z), np.imag(z)))
        feature = np.array([
            float(np.real(roots_sorted[0])),
            float(np.imag(roots_sorted[0])),
            float(np.real(roots_sorted[-1])),
            float(np.imag(roots_sorted[-1])),
            b0,
        ], dtype=float)
        return _ODEFitResult(
            order=2,
            params=np.array([a0, a1, b0], dtype=float),
            bic=self._bic(rss, len(d2x), 3),
            rss=rss,
            n_obs=len(d2x),
            cond_number=cond,
            is_stable=is_stable,
            feature=feature,
            roots=np.asarray(roots, dtype=complex),
        )

    def _fit_best(self, window: np.ndarray) -> Optional[_ODEFitResult]:
        x = np.asarray(window, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if len(x) < self.min_samples:
            return None
        x = self._smooth(x)
        candidates = [self._fit_order0(x)]
        fit1 = self._fit_order1(x)
        fit2 = self._fit_order2(x)
        if fit1 is not None:
            candidates.append(fit1)
        if fit2 is not None:
            candidates.append(fit2)

        valid: List[_ODEFitResult] = []
        for fit in candidates:
            if not np.all(np.isfinite(fit.params)) or not np.all(np.isfinite(fit.feature)):
                continue
            if fit.cond_number > self.cond_max:
                continue
            if self.stable_only and not fit.is_stable:
                continue
            valid.append(fit)
        if len(valid) == 0:
            return candidates[0] if len(candidates) > 0 else None
        by_order: Dict[int, _ODEFitResult] = {}
        for fit in valid:
            cur = by_order.get(int(fit.order))
            if cur is None or (fit.bic, fit.order) < (cur.bic, cur.order):
                by_order[int(fit.order)] = fit

        best_fit = min(valid, key=lambda fit: (fit.bic, fit.order))
        if self.prev_order is None:
            self.pending_order = None
            self.pending_order_hits = 0
            return best_fit

        current_fit = by_order.get(int(self.prev_order))
        if current_fit is None:
            self.pending_order = None
            self.pending_order_hits = 0
            return best_fit

        if int(best_fit.order) == int(self.prev_order):
            self.pending_order = None
            self.pending_order_hits = 0
            return current_fit

        improvement = float(current_fit.bic - best_fit.bic)
        if improvement <= self.order_switch_margin:
            self.pending_order = None
            self.pending_order_hits = 0
            return current_fit

        if self.pending_order is not None and int(self.pending_order) == int(best_fit.order):
            self.pending_order_hits += 1
        else:
            self.pending_order = int(best_fit.order)
            self.pending_order_hits = 1

        if self.pending_order_hits >= self.order_switch_patience:
            self.pending_order = None
            self.pending_order_hits = 0
            return best_fit

        return current_fit

    def _cluster_eps(self, order: int) -> float:
        if order == 0:
            return float(getattr(self.cfg, "ode_cluster_eps_order0", 0.8))
        if order == 1:
            return float(getattr(self.cfg, "ode_cluster_eps_order1", 1.2))
        return float(getattr(self.cfg, "ode_cluster_eps_order2", 1.2))

    def _normalize_features(self, rows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(rows, dtype=float)
        mu = X.mean(axis=0)
        sigma = np.maximum(X.std(axis=0), 1e-8)
        return (X - mu) / sigma, mu, sigma

    def _rebuild_clusters(self) -> None:
        self.cluster_centers = {0: [], 1: [], 2: []}
        self.cluster_rids = {0: [], 1: [], 2: []}
        self.next_rid = 0

        if len(self.window_history) < self.bootstrap_size:
            return

        for order in (0, 1, 2):
            fits = [fit for fit in self.window_history if fit.order == order]
            if len(fits) < self.cluster_min_samples:
                continue
            rows = [fit.feature for fit in fits]
            Xn, mu, sigma = self._normalize_features(rows)
            if not _HAS_DBSCAN:
                labels = np.zeros(len(fits), dtype=int)
            else:
                labels = _DBSCAN(
                    eps=self._cluster_eps(order),
                    min_samples=self.cluster_min_samples,
                ).fit_predict(Xn)
            keep_labels = sorted(label for label in set(labels.tolist()) if label >= 0)
            for label in keep_labels:
                members = [rows[i] for i, lab in enumerate(labels.tolist()) if lab == label]
                if len(members) == 0:
                    continue
                mem_arr = np.asarray(members, dtype=float)
                center_raw = np.median(mem_arr, axis=0)
                center_norm = (center_raw - mu) / sigma
                if self.next_rid >= self.max_regimes:
                    return
                self.cluster_centers[order].append(center_norm)
                self.cluster_rids[order].append(self.next_rid)
                self.next_rid += 1

    def _filter_feature(self, order: int, feature: np.ndarray) -> np.ndarray:
        f = np.asarray(feature, dtype=float).reshape(-1)
        if not self.use_feature_filter:
            return f

        order = int(order)
        if self.filter_reset_on_order_change and self.prev_order is not None and order != self.prev_order:
            self._filter_mean[order] = None
            self._filter_cov[order] = None

        mean_prev = self._filter_mean.get(order)
        cov_prev = self._filter_cov.get(order)
        dim = int(f.shape[0])
        q = self.filter_process_var
        r = self.filter_measure_var

        if mean_prev is None or cov_prev is None or mean_prev.shape[0] != dim:
            mean = f.copy()
            cov = np.eye(dim, dtype=float) * self.filter_init_var
        else:
            mean_pred = mean_prev
            cov_pred = cov_prev + np.eye(dim, dtype=float) * q
            S = cov_pred + np.eye(dim, dtype=float) * r
            K = cov_pred @ np.linalg.pinv(S)
            mean = mean_pred + K @ (f - mean_pred)
            cov = (np.eye(dim, dtype=float) - K) @ cov_pred

        self._filter_mean[order] = mean
        self._filter_cov[order] = cov
        return mean

    def _assign_cluster(self, fit: _ODEFitResult, feature_now: Optional[np.ndarray] = None) -> Optional[int]:
        order = int(fit.order)
        centers = self.cluster_centers.get(order, [])
        rids = self.cluster_rids.get(order, [])
        if len(centers) == 0 or len(rids) == 0:
            return None
        rows = [f.feature for f in self.window_history if f.order == order]
        rows.append(np.asarray(feature_now if feature_now is not None else fit.feature, dtype=float))
        Xn, _, _ = self._normalize_features(rows)
        f_now = Xn[-1]
        dists = np.array([np.linalg.norm(f_now - c) for c in centers], dtype=float)
        if len(dists) == 0:
            return None
        j = int(np.argmin(dists))
        if float(dists[j]) > self.assign_threshold:
            return None
        return int(rids[j])

    def _fallback_rid(self) -> int:
        if self.prev_rid is not None:
            return int(self.prev_rid)
        return 0

    def _update_and_get_regime(self, window: np.ndarray, residual: bool = False) -> int:
        fit = self._fit_best(window)
        if fit is None:
            return self._fallback_rid()

        self.window_history.append(fit)
        Wc = max(self.bootstrap_size * 4, int(getattr(self.cfg, "calib_window_size", 150)))
        if len(self.window_history) > Wc:
            self.window_history = self.window_history[-Wc:]

        self._steps_seen += 1
        need_rebuild = (
            len(self.window_history) == self.bootstrap_size
            or (len(self.window_history) > self.bootstrap_size and self._steps_seen % self.refit_every == 0)
        )
        if need_rebuild:
            self._rebuild_clusters()

        feature_now = self._filter_feature(int(fit.order), fit.feature)
        rid = self._assign_cluster(fit, feature_now=feature_now)
        if rid is None:
            rid = self._fallback_rid()

        rid = int(max(0, min(self.max_regimes - 1, rid)))
        self.prev_rid = rid
        self.prev_order = int(fit.order)
        return rid
