import torch
from src.lanczos.utils import *
# profiling
from tqdm import tqdm

class hessian:
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """Hessian computation module

        Args:
            model: the model that needs Hessain information
            criterion: the loss function
            data: a single batch of data, including inputs and their corresponding labels
            dataloader: the data loader including multiple of batches of data
            cuda: whether to use GPU for computation
        """
        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (
            data == None and dataloader != None
        )

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == "cuda":
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [
            torch.zeros(p.size()).to(device) for p in self.params
        ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0.0 for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def slow_lanczos(self, iter, seed=0):
        """ Computes the eigenvalues using the Slow Lanczos algorithm (Papyan's implementation; with full reorthogonalization) 

        Args:
            iter: number of iterations (should be set to number of parameters for full spectrum)
            seed: random seed for reproducibility

        Returns:
            eigen_list: list of eigenvalues
            weight_list: list of weights
        """
        # initialization
        alpha_list = []
        beta_list = []
        ############### Lanczos algorithm ################
        for i in tqdm(range(iter)):
            if i == 0:
                if seed != 0:
                    torch.manual_seed(seed)
                # old: v_1 = normalization(torch.randn(self.size, device=self.device))
                v_1 = generate_random_vector(self.params)
                v_1 = normalization(v_1)
                v_list = [v_1]
                # old: w = self.matvec(v_1)
                if self.full_dataset:
                    _, w = self.dataloader_hv_product(v_1)
                else:
                    w = hessian_vector_product(self.gradsH, self.params, v_1)

            else:
                # old: w = self.matvec(v_list[-1]) - beta_list[-1] * v_list[-2]
                if self.full_dataset:
                    _, w = self.dataloader_hv_product(v_list[-1])
                else:
                    w = hessian_vector_product(self.gradsH, self.params, v_list[-1])
                w = group_add(w, v_list[-2], alpha=-beta_list[-1])
            # old: alpha = torch.dot(v_list[-1], w)
            alpha = group_product(w, v_list[-1])
            # old: w = w - alpha * v_list[-1]
            w = group_add(w, v_list[-1], alpha=-alpha)
            # reorthogonalization
            w = orthogonal(w, v_list)
            # old: beta = torch.sqrt(torch.dot(w, w))
            beta = torch.sqrt(group_product(w, w))
            v = normalization(w)
            v_list.append(v)
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())
        # form tridiagonal matrix and compute eigenvalues
        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())

    def fast_lanczos(self, iter, seed=0):
        """ Computes the eigenvalues using the Fast Lanczos algorithm (Papyan's implementation; without reorthogonalization)

        Args:
            iter: number of iterations (should be set to number of parameters for full spectrum)
            seed: random seed for reproducibility

        Returns:
            eigen_list: list of eigenvalues
            weight_list: list of weights
        """
        # initialization
        alpha_list = []
        beta_list = []
        ############### Lanczos algorithm ################
        for i in tqdm(range(iter)):
            if i == 0:
                if seed != 0:
                    torch.manual_seed(seed)
                # old: v = normalization(torch.randn(self.size, device=self.device))
                v = generate_random_vector(self.params)
                v = normalization(v)
                # old: v_next = self.matvec(v)
                if self.full_dataset:
                    _, v_next = self.dataloader_hv_product(v)
                else:
                    v_next = hessian_vector_product(self.gradsH, self.params, v)
            else:
                # old: v_next = self.matvec(v) - beta_list[-1] * v_prev
                if self.full_dataset:
                    _, v_next = self.dataloader_hv_product(v)
                else:
                    v_next = hessian_vector_product(self.gradsH, self.params, v)
                v_next = group_add(v_next, v_prev, alpha=-beta_list[-1])
            # old: alpha = torch.dot(v_next, v)
            alpha = group_product(v_next, v)
            # old: v_next = v_next - alpha * v
            v_next = group_add(v_next, v, alpha=-alpha)
            # old: beta = torch.sqrt(torch.dot(v_next, v_next))
            beta = torch.sqrt(group_product(v_next, v_next))
            v_next = normalization(v_next)
            v_prev = v
            v = v_next
            alpha_list.append(alpha.cpu().item())
            beta_list.append(beta.cpu().item())
        # form tridiagonal matrix and compute eigenvalues
        T = form_tridiagonal_mtx(alpha_list, beta_list, self.device)
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        return list(eigen_list.cpu().numpy()), list(weight_list.cpu().numpy())

    def stochastic_lanczos_quadrature(self, method, iter=100, n_v=1, seed=0):
        eigen_list_full = []
        weight_list_full = []
        if method == "slow_papyan":
            for i in range(n_v):
                eigen_list, weight_list = self.slow_lanczos_papyan(iter, seed)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        elif method == "fast_papyan":
            for i in range(n_v):
                eigen_list, weight_list = self.fast_lanczos_papyan(iter, seed)
                eigen_list_full.append(eigen_list)
                weight_list_full.append(weight_list)
        else:
            raise ValueError("Invalid method")
        return eigen_list_full, weight_list_full
