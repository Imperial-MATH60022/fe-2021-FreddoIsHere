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


def assemble(fs, f):
    Q = gauss_quadrature(fs.mesh.cell, fs.element.degree + 2)
    phi = fs.element.tabulate(Q.points)  # (points, nodes)
    phi_grad = fs.element.tabulate(Q.points, grad=True)  # (points, nodes, dim)

    # Assemble F block and zero block
    F = np.zeros(fs.mesh.cell_vertices.shape[0]*fs.element.node_count)
    for c in range(fs.mesh.cell_vertices.shape[0]):
        cell_nodes = fs.cell_nodes[c, :]
        J = fs.mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))
        values = f.values[cell_nodes].reshape(-1, 2)
        # multiply quadrature weights with the point indices of phi
        weighted_phi = np.einsum("i, ijk->ijk", Q.weights, phi)
        # multiply each function value with the correct node values
        values_phi = np.einsum("ij, pij->pij", values, phi)
        F[cell_nodes] += (np.einsum("pnd, pjk-> nk", weighted_phi, values_phi) * detJ).flatten()

    F = F.reshape(-1, 2)
    zero_block = np.zeros_like(F)



    return None, None


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
    fe = LagrangeElement(mesh.cell, 2)
    vfe = VectorFiniteElement(fe)
    fs = FunctionSpace(mesh, vfe)
    analytic_answer = Function(fs)
    # orthogonal gradient to γ?
    analytic_answer.interpolate(lambda x: x)
    if analytic:
        return analytic_answer, 0.0

    f = Function(fs)
    f.interpolate(lambda x: (2 * pi * (1 - cos(2 * pi * x[0])) * sin(2 * pi * x[1]),
                             -2 * pi * (1 - cos(2 * pi * x[1])) * sin(2 * pi * x[0])))

    A, l = assemble(fs, f)

    u = Function(fs)

    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    error = verrornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    return u, error


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
