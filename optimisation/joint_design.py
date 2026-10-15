"""Feasibility-first, learning- and participation-aware RIS/client scheduler.

The scheduler alternates between (1) selecting feasible clients using their
recent validation utility and successful-upload debt, and (2) tuning the RIS
for the selected clients. It is a deterministic heuristic for a mixed-integer
non-convex problem, not a globally optimal solver.
"""

import numpy as np


class JointDeviceRISOptimizer:
    """Jointly schedule clients and optimize a shared RIS configuration."""

    def __init__(self, channel, ris_optimizer, config: dict):
        self.channel = channel
        self.ris_optimizer = ris_optimizer
        selection_cfg = config.get('selection', {})
        fl_cfg = config.get('fl', {})

        self.latency_budget = float(selection_cfg.get('latency_budget_ms', 100.0))
        self.energy_budget = float(selection_cfg.get('energy_budget_mj', 50.0))
        self.max_iters = int(selection_cfg.get('max_alternating_iters', 5))
        self.tx_power_dbm = float(selection_cfg.get('tx_power_dbm', 23.0))
        self.cpu_ghz = float(selection_cfg.get('computation_capacity_ghz', 1.0))
        self.cycles_per_sample = float(selection_cfg.get('cycles_per_sample', 20.0))
        self.computation_energy_per_sample_mj = float(
            selection_cfg.get('computation_energy_per_sample_mj', 0.001))
        self.local_epochs = int(fl_cfg.get('local_epochs', 1))
        configured_clients = int(fl_cfg.get('num_clients', 20))
        cap = selection_cfg.get('max_participants')
        if cap is None:
            cap = int(fl_cfg.get('fraction_fit', 0.5) * configured_clients)
        self.max_clients = max(0, min(configured_clients, int(cap)))
        self.model_size_bits = selection_cfg.get(
            'model_size_bits', config.get('system', {}).get('model_size_bits'))
        self.utility_weight = float(selection_cfg.get('learning_utility_weight', 0.7))
        self.fairness_weight = float(selection_cfg.get('participation_debt_weight', 0.3))
        self.utility_momentum = float(selection_cfg.get('utility_ema_momentum', 0.8))
        self.utility_scores = np.zeros(configured_clients, dtype=float)
        self.success_counts = np.zeros(configured_clients, dtype=float)
        self.selection_counts = np.zeros(configured_clients, dtype=float)

    def observe_round(self, selected_indices, successful_indices,
                      utility_gains=None):
        """Update participation history and validation-utility estimates."""
        for idx in selected_indices:
            self.selection_counts[int(idx)] += 1.0
        successful = set(int(i) for i in successful_indices)
        gains = utility_gains or {}
        for idx in successful:
            self.success_counts[idx] += 1.0
            if idx in gains:
                old = self.utility_scores[idx]
                new = float(gains[idx])
                self.utility_scores[idx] = (
                    self.utility_momentum * old
                    + (1.0 - self.utility_momentum) * new)

    def optimize(self, h_br: np.ndarray, h_ru_list: list[np.ndarray],
                 h_d_list: list[np.ndarray], client_data_sizes: list[int],
                 client_comp_capacities: list[float]) -> tuple[list[int], np.ndarray]:
        """Return a feasible participant set and RIS phases.

        Empty selection is intentional when no client satisfies both hard
        budgets. Callers must treat it as an idle round, not force a violation.
        """
        n_clients = len(h_ru_list)
        if n_clients == 0 or self.max_clients == 0:
            return [], self.ris_optimizer.random_phases()
        model_bits = int(self.model_size_bits or 1_000_000)
        tx_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        priorities = self._priorities(n_clients)

        phi = self.ris_optimizer.optimize_sum_snr(
            h_br, h_ru_list, h_d_list, user_weights=priorities)
        selected = []
        for _ in range(max(1, self.max_iters)):
            rates, latencies, energies = self._resource_estimates(
                h_br, h_ru_list, h_d_list, phi, client_data_sizes,
                client_comp_capacities, model_bits, tx_w)
            feasible = [i for i in range(n_clients)
                        if latencies[i] <= self.latency_budget
                        and energies[i] <= self.energy_budget]
            updated = self._greedy_feasible_selection(feasible, priorities)
            if updated == selected and selected:
                break
            selected = updated
            if not selected:
                return [], phi
            selected_ru = [h_ru_list[i] for i in selected]
            selected_d = [h_d_list[i] for i in selected]
            phi = self.ris_optimizer.optimize_sum_snr(
                h_br, selected_ru, selected_d,
                user_weights=priorities[selected])

        # Recheck feasibility using the final RIS phases.
        rates, latencies, energies = self._resource_estimates(
            h_br, h_ru_list, h_d_list, phi, client_data_sizes,
            client_comp_capacities, model_bits, tx_w)
        selected = [i for i in selected
                    if latencies[i] <= self.latency_budget
                    and energies[i] <= self.energy_budget]
        return selected, phi

    def _priorities(self, n_clients):
        utility = self._rank_scale(self.utility_scores[:n_clients])
        # Clients with fewer successful uploads receive a larger debt score.
        debt = self._rank_scale(-self.success_counts[:n_clients])
        weights = (self.utility_weight * utility
                   + self.fairness_weight * debt)
        return np.maximum(weights, 1e-6)

    @staticmethod
    def _rank_scale(values):
        values = np.asarray(values, dtype=float)
        if values.size <= 1 or np.allclose(values, values[0]):
            return np.ones(values.size, dtype=float)
        # Average ranks keep tied utilities/debts tied instead of introducing
        # an arbitrary client-id preference through the sorting algorithm.
        order = np.argsort(values, kind='stable')
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks / max(values.size - 1, 1)

    def _resource_estimates(self, h_br, h_ru_list, h_d_list, phi,
                            client_data_sizes, client_comp_capacities,
                            model_bits, tx_w):
        rates, latencies, energies = [], [], []
        for k in range(len(h_ru_list)):
            h_eff = self.channel.compute_effective_channel(
                h_br, h_ru_list[k], h_d_list[k], phi)
            snr = float(np.real(self.channel.compute_snr(h_eff, self.tx_power_dbm)))
            rate = max(float(self.channel.compute_achievable_rate(max(snr, 0.0))), 1e-12)
            data_size = int(client_data_sizes[k])
            capacity = float(client_comp_capacities[k])
            cpu_hz = capacity if capacity > 1e6 else self.cpu_ghz * 1e9
            compute_s = (data_size * self.local_epochs * self.cycles_per_sample) / max(cpu_hz, 1e-12)
            comm_s = model_bits / rate
            rates.append(rate)
            latencies.append((compute_s + comm_s) * 1000.0)
            compute_mj = data_size * self.local_epochs * self.computation_energy_per_sample_mj
            energies.append(tx_w * comm_s * 1000.0 + compute_mj)
        return rates, latencies, energies

    def _greedy_feasible_selection(self, feasible, priorities):
        ranked = sorted(feasible,
                        key=lambda i: (priorities[i], -self.success_counts[i], -i),
                        reverse=True)
        return [int(i) for i in ranked[:self.max_clients]]
