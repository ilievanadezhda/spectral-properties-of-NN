import torch
from tqdm import tqdm
from utils import normalization, orthnormal, form_tridiagonal_mtx


class matrix:
    def __init__(self, eigenvalues):
        """
        eigenvalues: a list of eigenvalues
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = len(eigenvalues)
        self.eigenvalues = eigenvalues.to(self.device)
        self.diag_mtx = torch.diag(self.eigenvalues).to(self.device)
        self.u_vectors = [
            normalization(torch.randn(self.size, device=self.device))
            for _ in range(self.size)
        ]

    def get_matrix(self):
        """
        compute A = U*D*U^T
        """
        U = torch.eye(self.size, device=self.device)
        for u in self.u_vectors:
            U = torch.matmul(
                U, torch.eye(self.size, device=self.device) - 2 * torch.outer(u, u)
            )
        return torch.matmul(U, torch.matmul(self.diag_mtx, U.T))

    def matvec(self, h):
        """
        compute matrix-vector product
        """
        for u in self.u_vectors:
            h = h - 2 * torch.dot(h, u) * u
        h = torch.matmul(self.diag_mtx, h)
        for u in reversed(self.u_vectors):
            h = h - 2 * torch.dot(h, u) * u
        return h

    def matvec_full(self, h):
        """
        compute matrix-vector product using the full matrix
        """
        return torch.matmul(self.get_matrix(), h)

    def slow_lanczos_pyhessian(self, iter, seed=0):
        """
        compute the eigenvalues using slow lanczos algorithm (PyHessian implementation; with full reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        # generate random vector
        if seed != 0:
            torch.manual_seed(seed)
        v = normalization(torch.randn(self.size, device=self.device))
        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        for i in range(iter):
            w_prime = torch.zeros(self.size, device=self.device)
            if i == 0:
                w_prime = self.matvec(v)
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = w_prime - alpha * v
                w_list.append(w)
            else:
                beta = torch.sqrt(torch.dot(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.0:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    print("beta is zero, generate a new vector, iter = ", i)
                    w = torch.randn(self.size, device=self.device)
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                w_prime = self.matvec(v)
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = w_prime - alpha * v - beta * v_list[-2]

        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)

        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())

    def slow_lanczos_papyan(self, iter, seed=0):
        """
        compute the eigenvalues using slow lanczos algorithm (Papyan implementation; with full reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        alpha_list = []
        beta_list = []
        for i in range(iter):
            if i == 0:
                if seed != 0:
                    torch.manual_seed(seed)
                v_1 = normalization(torch.randn(self.size, device=self.device))
                v_list = [v_1]
                w = self.matvec(v_1)
            else:
                w = self.matvec(v_list[-1]) - beta_list[-1] * v_list[-2]
            alpha = torch.dot(v_list[-1], w)
            w = w - alpha * v_list[-1]
            # reorthogonalization
            for v in v_list:
                w = w - torch.dot(w, v) * v
            beta = torch.sqrt(torch.dot(w, w))
            v = w / beta
            v_list.append(v)
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())

        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())
        
    def slow_lanczos_demmel(self, iter, seed=0):
        """
        compute the eigenvalues using slow lanczos algorithm (Demmel implementation; double full reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        # generate random vector
        if seed != 0:
            torch.manual_seed(seed)
        v = normalization(torch.randn(self.size, device=self.device))
        v_list = [torch.zeros(self.size), v]
        # empty lists for storing alpha and beta values
        alpha_list = []
        beta_list = [0]
        for i in range(iter):
            w = self.matvec(v_list[-1])
            alpha = torch.dot(v_list[-1], w)
            # reorthogonalization
            for v in v_list:
                w = w - torch.dot(w, v) * v
            # double reorthogonalization
            for v in v_list:
                w = w - torch.dot(w, v) * v
            beta = torch.sqrt(torch.dot(w, w))
            if beta == 0:
                raise ValueError("beta is zero!")
            v = w / beta
            v_list.append(v)
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())

        # remove the first element of beta_list
        beta_list = beta_list[1:]
        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())

    def fast_lanczos_papyan(self, iter, seed=0):
        """
        compute the eigenvalues using fast lanczos algorithm (Papyan implementation; no reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        alpha_list = []
        beta_list = []
        for i in range(iter):
            if i == 0:
                if seed != 0:
                    torch.manual_seed(seed)
                v = normalization(torch.randn(self.size, device=self.device))
                v_next = self.matvec(v)
            else:
                v_next = self.matvec(v) - beta_list[-1] * v_prev
            alpha = torch.dot(v_next, v)
            v_next = v_next - alpha * v
            beta = torch.sqrt(torch.dot(v_next, v_next))
            v_next = v_next / beta
            v_prev = v
            v = v_next
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())

        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())
    
    def fast_lanczos_demmel(self, iter, seed=0):
        """
        compute the eigenvalues using slow lanczos algorithm (Demmel implementation; no reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        # generate random vector
        if seed != 0:
            torch.manual_seed(seed)
        v = normalization(torch.randn(self.size, device=self.device))
        v_list = [0, v]
        # empty lists for storing alpha and beta values
        alpha_list = []
        beta_list = [0]
        for i in range(iter):
            w = self.matvec(v_list[-1])
            alpha = torch.dot(v_list[-1], w)
            w = w - alpha * v_list[-1] - beta_list[-1] * v_list[-2]
            beta = torch.sqrt(torch.dot(w, w))
            if beta == 0:
                raise ValueError("beta is zero!")
            v = w / beta
            v_list.append(v)
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())
        
        # remove the first element of beta_list
        beta_list = beta_list[1:]
        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())
    
    def selective_lanczos_demmel(self, iter, seed=0):
        """ 
        compute the eigenvalues using selective lanczos algorithm (Demmel implementation; with selective reorthogonalization)
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        # generate random vector
        if seed != 0:
            torch.manual_seed(seed)
        v = normalization(torch.randn(self.size, device=self.device))
        v_list = [0, v]
        # empty lists for storing alpha and beta values
        alpha_list = []
        beta_list = [0]
        for i in range(iter):
            w = self.matvec(v_list[-1])
            alpha = torch.dot(v_list[-1], w)
            w = w - alpha * v_list[-1] - beta_list[-1] * v_list[-2]
            # TODO: selective reorthogonalization
            beta = torch.sqrt(torch.dot(w, w))
            if beta == 0:
                raise ValueError("beta is zero!")
            v = w / beta
            v_list.append(v)
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())
            
        # remove the first element of beta_list
        beta_list = beta_list[1:]
        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())

    def stochastic_lanczos_quadrature(self, method, iter=100, n_v=1):
        eigen_list_full = []
        weight_list_full = []
        if method == "slow_pyhessian":
            for i in range(n_v):
                eigen_list, weight_list = self.slow_lanczos_pyhessian(iter)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        elif method == "slow_papyan":
            for i in range(n_v):
                eigen_list, weight_list = self.slow_lanczos_papyan(iter)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        elif method == "slow_demmel":
            for i in range(n_v):
                eigen_list, weight_list = self.slow_lanczos_demmel(iter)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        elif method == "fast_papyan":
            for i in range(n_v):
                eigen_list, weight_list = self.fast_lanczos_papyan(iter)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        elif method == "fast_demmel":
            for i in range(n_v):
                eigen_list, weight_list = self.fast_lanczos_demmel(iter)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        else:
            raise ValueError("Invalid method")
        return eigen_list_full, weight_list_full
