import torch


class Scaler(object):
    """Generate scale and offset based on running mean and stddev along axis=0

    offset = running mean
    scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = torch.zeros(obs_dim)
        self.means = torch.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        x = x.reshape(-1)
        """Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = torch.mean(x, axis=0)
            self.vars = torch.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = torch.var(x, axis=0)
            new_data_mean = torch.mean(x, axis=0)
            new_data_mean_sq = torch.square(new_data_mean)
            new_means = ((self.means * self.m) +
                         (new_data_mean * n)) / (self.m + n)
            self.vars = (
                (self.m * (self.vars + torch.square(self.means)))
                + (n * (new_data_var + new_data_mean_sq))
            ) / (self.m + n) - torch.square(new_means)
            self.vars = self.vars.clamp(min=0.0)
            self.means = new_means
            self.m += n

    def get(self):
        """returns 2-tuple: (scale, offset)"""
        return 1 / ((torch.sqrt(self.vars) + 0.1) / 3), self.means

    def norm(self, x):
        scale, offset = self.get()
        return (x - offset) * scale


class Scaler_Moving_Aver(object):
    """Generate scale and offset based on running mean and stddev along axis=0

    offset = running mean
    scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim, aver_rate=0.9):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = torch.zeros(obs_dim)
        self.means = torch.zeros(obs_dim)
        self.first_pass = True
        self.aver_rate = aver_rate

    def update(self, x):
        x = x.reshape(-1)
        """Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = torch.mean(x, axis=0)
            self.vars = torch.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            new_data_var = torch.var(x, axis=0)
            new_data_mean = torch.mean(x, axis=0)

            self.means = (1 - self.aver_rate) * self.means + \
                self.aver_rate * new_data_mean
            self.vars = (1 - self.aver_rate) * self.vars + \
                self.aver_rate * new_data_var

    def get(self):
        """returns 2-tuple: (scale, offset)"""
        return 1 / ((torch.sqrt(self.vars) + 0.1) / 3), self.means

    def norm(self, x):
        scale, offset = self.get()
        return (x - offset) * scale
