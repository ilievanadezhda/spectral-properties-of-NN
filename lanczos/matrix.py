import torch
from utils import * 

class matrix():
    def __init__(self, eigenvalues):
        """
        eigenvalues: a list of eigenvalues
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = len(eigenvalues)
        self.eigenvalues = eigenvalues.to(self.device)
        self.diag_mtx = torch.diag(self.eigenvalues).to(self.device)
        self.u_vectors = [normalization(torch.randn(self.size, device=self.device)) for _ in range(self.size)]

    def get_matrix(self):
        """
        compute A = U*D*U^T
        """
        U = torch.eye(self.size, device=self.device)
        for u in self.u_vectors:
            U = torch.matmul(U, torch.eye(self.size, device=self.device) - 2 * torch.outer(u, u))
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
        return torch.matmul(self.compute_self(), h)
    
    def slow_lanczos(self, iter):
        """
        compute the eigenvalues using slow lanczos algorithm
        iter: number of iterations (should be set to number of parameters for full spectrum)
        """
        device = self.device
        # generate random vector
        v = normalization(torch.randn(self.size, device=device))
        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        for i in range(iter):
            w_prime = torch.zeros(self.size, device=device)
            if i == 0:
                w_prime = self.matvec(v)
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = w_prime - alpha * v
                w_list.append(w)
            else:
                beta = torch.sqrt(torch.dot(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).to(device) for p in self.params]
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                w_prime = self.matvec(v)
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = w_prime - alpha * v
                w = w_tmp - beta * v_list[-2]

        T = torch.zeros(iter, iter).to(device)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        eigenvalues, _ = torch.linalg.eig(T)
        eigen_list = eigenvalues.real

        return list(eigen_list.cpu().numpy())
    
    def fast_lanczos(self, iter):
        pass

    def stochastic_lanczos_quadrature(self, method, iter, n_v):
        pass