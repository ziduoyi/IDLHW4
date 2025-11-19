import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

        self.x = None
        self.hidden = None
        self.r = None
        self.z = None
        self.n = None
        self.u_r = None
        self.u_z = None
        self.u_n = None
        self.q = None

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        u_r = self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh
        r = self.r_act.forward(u_r)

        u_z = self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh
        z = self.z_act.forward(u_z)

        q = self.Wnh @ h_prev_t + self.bnh

        u_n = self.Wnx @ x + self.bnx + r * q
        n = self.h_act.forward(u_n)

        h_t = (1.0 - z) * n + z * h_prev_t

        self.r, self.z, self.n = r, z, n
        self.u_r, self.u_z, self.u_n = u_r, u_z, u_n
        self.q = q

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        x = self.x
        h_prev = self.hidden
        r, z, n = self.r, self.z, self.n
        q = self.q

        dn = delta * (1.0 - z)
        dz = delta * (h_prev - n)
        dh_prev_mix = delta * z

        du_n = dn * (1.0 - n**2)

        dr = du_n * q
        dq = du_n * r

        du_z = dz * z * (1.0 - z)

        du_r = dr * r * (1.0 - r)

        self.dWnx += np.outer(du_n, x)
        self.dWzx += np.outer(du_z, x)
        self.dWrx += np.outer(du_r, x)

        self.dWnh += np.outer(dq, h_prev)
        self.dWzh += np.outer(du_z, h_prev)
        self.dWrh += np.outer(du_r, h_prev)

        self.dbnx += du_n
        self.dbzx += du_z
        self.dbrx += du_r

        self.dbnh += dq
        self.dbzh += du_z
        self.dbrh += du_r

        dx = du_n @ self.Wnx + du_z @ self.Wzx + du_r @ self.Wrx

        dh_prev_t = dh_prev_mix + du_z @ self.Wzh + du_r @ self.Wrh + dq @ self.Wnh


        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
