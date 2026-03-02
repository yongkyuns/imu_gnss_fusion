import numpy as np


class RLS:
    def __init__(
        self,
        initial_state,
        initial_covariance_diagonal,
        measurement_uncertainty_R_scalar,
        forgetting_factor,
    ):
        if len(initial_state) != 3:
            raise ValueError("Initial state for RLS must be a 3-element vector.")
        self.x = np.array(initial_state, dtype=np.float32)

        N = 3
        self.P = np.eye(N, dtype=np.float32) * np.float32(initial_covariance_diagonal)
        self.R_matrix = np.eye(N, dtype=np.float32) * np.float32(
            initial_covariance_diagonal
        )
        self.forgetting_factor = np.float32(forgetting_factor)

    def update(self, z):
        if len(z) != 3:
            raise ValueError("Measurement for RLS must be a 3-element vector.")
        z = np.array(z, dtype=np.float32)
        P_minus = self.P / self.forgetting_factor

        innovation = z - self.x

        P_minus_diag_elements = np.diag(P_minus)
        R_diag_elements = np.diag(self.R_matrix)
        S_diag_elements = P_minus_diag_elements + R_diag_elements

        epsilon = np.finfo(S_diag_elements.dtype).eps
        K_diag_elements = P_minus_diag_elements / (S_diag_elements + epsilon)
        K = np.diag(K_diag_elements)

        self.x = self.x + K @ innovation

        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = (I - K) @ P_minus

        return self.x
