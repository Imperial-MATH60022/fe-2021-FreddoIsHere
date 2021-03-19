"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from argparse import ArgumentParser
from fe_utils import *
import numpy as np
from numpy import sin, cos, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from fe_utils.finite_elements import VectorFiniteElement


def verrornorm(analytic_answer, u):
    return 0.0


def assemble(fs_u, fs_p, f):
    n = fs_p.mesh.entity_counts[0]
    m = 2*(n + fs_u.mesh.entity_counts[1])
    F = np.zeros(m)
    B = sp.lil_matrix((n, m))
    A = sp.lil_matrix((m, m))

    Q_u = gauss_quadrature(fs_u.mesh.cell, fs_u.element.degree + 2)
    phi = fs_u.element.tabulate(Q_u.points)
    phi_grad = fs_u.element.tabulate(Q_u.points, grad=True)
    psi = fs_p.element.tabulate(Q_u.points)

    for c in range(fs_u.mesh.entity_counts[-1]):
        cell_nodes_u = fs_u.cell_nodes[c, :]
        cell_nodes_p = fs_p.cell_nodes[c, :]
        J = fs_u.mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))
        inv_J = np.linalg.inv(J)
        values = f.values[cell_nodes_u].reshape(-1, 2)
        """ F-assembly """
        # multiply quadrature weights with the point indices of phi
        weighted_phi = np.einsum("p, pjk->pjk", Q_u.weights, phi)
        # multiply each function value with the correct node values
        values_phi = np.einsum("ij, pij->pij", values, phi)
        F[cell_nodes_u] += (np.einsum("pnd, pjk-> nk", weighted_phi, values_phi) * detJ).flatten()
        """ A-assembly """
        # J^{−T} ∇_XΦ_i(X)
        inv_J_phi_grad = np.einsum("dk, pnkj->pnkj", inv_J.T, phi_grad)
        # J^{−T} ∇_XΦ_i(X) + (J^{−T} ∇_XΦ_i(X))^T
        inv_J_phi_grad2 = inv_J_phi_grad + inv_J_phi_grad.swapaxes(2, 3)
        # multiply quadrature weights with the point indices of inv_J_phi_grad2
        weighted_inv_J_phi_grad2 = np.einsum("p, pnkj->pnkj", Q_u.weights, inv_J_phi_grad2)
        # Collapse the vector axes into one
        weighted_inv_J_phi_grad2 = weighted_inv_J_phi_grad2.reshape((phi_grad.shape[0], -1, phi_grad.shape[-1]))
        inv_J_phi_grad2 = inv_J_phi_grad2.reshape((phi_grad.shape[0], -1, phi_grad.shape[-1]))
        # sum over quadrature weights and multiply
        sum = np.einsum("pnd, pik->ni", weighted_inv_J_phi_grad2, inv_J_phi_grad2)
        A[np.ix_(cell_nodes_u, cell_nodes_u)] += sum*detJ
        """ B-assembly """
        # Multiply psi basis by J^{−T} ∇_XΦ_i(X)
        sum = np.einsum("pn, pik->ni", psi, weighted_inv_J_phi_grad2)
        B[np.ix_(cell_nodes_p, cell_nodes_u)] = sum*detJ

    lhs = sp.bmat([[A, B.T], [B, None]])
    rhs = np.hstack((F, np.zeros(n)))
    # enforcing zero-Dirichlet on V
    u_bound_nodes = boundary_nodes(fs_u)
    lhs[u_bound_nodes] = 0
    rhs[u_bound_nodes] = 0
    # enforcing zero-Dirichlet at arbitrary node
    arb_node = np.random.randint(low=m, high=m+n)
    lhs[arb_node] = 0
    rhs[arb_node] = 0

    return lhs, rhs


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given resolution. It
    should return both the solution :class:`~fe_utils.function_spaces.Function` and
    the :math:`L^2` error in the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """

    mesh = UnitSquareMesh(resolution, resolution)
    fe_p = LagrangeElement(mesh.cell, 1)
    fe_u = LagrangeElement(mesh.cell, 2)
    fe_u = VectorFiniteElement(fe_u)
    fs_p = FunctionSpace(mesh, fe_p)
    fs_u = FunctionSpace(mesh, fe_u)

    f = Function(fs_u)
    f.interpolate(lambda x: (2 * pi * (1 - cos(2 * pi * x[0])) * sin(2 * pi * x[1]),
                             -2 * pi * (1 - cos(2 * pi * x[1])) * sin(2 * pi * x[0])))

    lhs, rhs = assemble(fs_u, fs_p, f)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_mastery(resolution, analytic, plot_error)

    u.plot()
