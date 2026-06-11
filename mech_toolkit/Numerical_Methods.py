import logging
import os
from typing import List, Tuple, Callable, Optional


# -- Logging Configuration ---------------------------------------------
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "numerical_methods.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(_LOG_FILE, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_file_handler)

class Interpolation:
    """
    A class to perform various interpolation methods on a given set of data points.
    Attributes:
        :x_values (list): A list of x-coordinates of the data points.
        :y_values (list): A list of y-coordinates of the data points.
        :n (int): The number of data points.
    Methods:
        :lagrange(x): Computes the Lagrange interpolation polynomial at a given point x.
        :newton(x): Computes the Newton interpolation polynomial at a given point x.
        :linear_spline(x): Computes the linear spline interpolation at a given point x.
        :cubic_spline(x): Computes the cubic spline interpolation at a given point x.
    Raises:
        ValueError: If the input data points are invalid (e.g., different lengths, non-distinct x_values).
    """



    def __init__(self, x_values: List[float], y_values: List[float]):
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length.")    # Ensure both lists have the same length
        if len(set(x_values)) != len(x_values): # Check for distinct x_values
            raise ValueError("x_values must be distinct for interpolation.")
        paired = sorted(zip(x_values, y_values), key=lambda p:p[0]) # Sort the data points by x_values

        self.x_values = [p[0] for p in paired]
        self.y_values = [p[1] for p in paired]
        self.n = len(x_values)
        self._cubic_spline_coefficients: Optional[Tuple[List[float], List[float], List[float]]] = None # Cache for cubic spline coefficients

    def lagrange_polynomial_str(self) -> str:
        """
        Return a string representation of the lagrange interpolation polynomial.
        """
        terms: List[str] = []
        for i in range(self.n):
            if self.y_values[i] == 0:
                continue
            basis_part: List[str] = []
            for j in range(self.n):
                if j != i:
                    denom = self.x_values[i] - self.x_values[j]
                    basis_part.append(f"(x - {self.x_values[j]})/{denom:.6g}")
            basis = " * ".join(basis_part)
            terms.append(f"{self.y_values[i]:.6g} * {basis}")
        poly_str = " + ".join(terms) if terms else "0"
        return f"P(x) = {poly_str}"
    
    def newton_polynomial_str(self) -> str:
        """
        Return a string representation of the Newton interpolation polynomial.
        """
        coef = self._divided_differences()
        terms: List[str] = [f"{coef[0][0]:.6g}"]
        for i in range(1, self.n):
            if coef[0][i] == 0:
                continue
            factors = " * ".join(f"( x - {self.x_values[j]:.6g})" for j in range(i))
            terms.append(f"{coef[0][i]:.6g} * {factors}")
        poly_str = " + ".join(terms) if terms else "0"
        return f"P(x) = {poly_str}"
    
    def linear_spline_str(self) -> str:
        """
        Return a string representation of the linear spline (piecewise)
        """
        pieces: List[str] = []
        for i in range(self.n - 1):
            slope = (self.y_values[i + 1] - self.y_values[i]) / (self.x_values[i + 1] - self.x_values[i])
            intercept = self.y_values[i] - slope * self.x_values[i]
            pieces.append(
                f" S{i}(x) = {slope:.6g}*x + {intercept:.6g}, "
                f"x in [{self.x_values[i]:.6g}, {self.x_values[i + 1]:.6g}]"
            )
        return "Linear Spline:\n" + "\n".join(pieces)
    
    def cubic_spline_str(self) -> str:
        """
        Return a string representation of the cubic spline (piecewise)
        """
        b, c, d = self._compute_cubic_spline_coefficients()
        pieces: List[str] = []
        for i in range(self.n -1):
            xi = self.x_values[i]
            pieces.append(
                f" S{i}(x) = {self.y_values[i]:.6g} + {b[i]:.6g}*(x - {xi:.6g})"
                f"+ {c[i]:.6g}*(x - {xi:.6g})^2 + {d[i]:.6g}*(x - {xi:.6g})^3, "
                f"x in [{self.x_values[i]:.6g}, {self.x_values[i + 1]:.6g}]"
            )
        return "Cubic Spline:\n" + "\n".join(pieces)


    def lagrange(self, x: float) -> float:

        """
        Compute the Lagrange interpolation polynomial at a given point x.
        Args:
            x: The point at which to evaluate the interpolation polynomial.
        Returns:
            The value of the interpolation polynomial at x.
        Raises:
            ValueError: If the x_values are not distinct.
        """
        logger.info(f"Lagrange interpolation called with x={x}")
        result = 0.0
        for i in range(self.n):
            term = self.y_values[i]
            for j in range(self.n):
                if j != i:
                    term *= (x - self.x_values[j]) / (self.x_values[i] - self.x_values[j])
            result += term
        logger.info(f"Lagrange result: P({x}) = {result}")
        logger.debug(f"Lagrange polynomial: {self.lagrange_polynomial_str()}")
        return result
    
    def _divided_differences(self) -> list:
        """Compute the divided differences table."""
        n = self.n
        coef:List[List[float]] = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            coef[i][0] = self.y_values[i]
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (self.x_values[i + j] - self.x_values[i]) 
        return coef

    def newton(self, x: float) -> float:
        """
        Compute the Newton interpolation polynomial at a given point x.
        Args:
            x: The point at which to evaluate the interpolation polynomial.
        Returns:
            The value of the interpolation polynomial at x.
        Raises:
            ValueError: If the x_values are not distinct.
        """ 
        logger.info(f"Newton interpolation called with x={x}")
        coef = self._divided_differences()
        result = coef[0][0]
        term = 1.0
        for i in range(1, self.n):
            term *= (x - self.x_values[i - 1])
            result += coef[0][i] * term
        logger.info(f"Newton result: P({x}) = {result}")
        logger.debug(f"Newton polynomial: {self.newton_polynomial_str()}")
        return result

    def linear_spline(self, x: float) -> float:
        """
        Compute the linear spline interpolation at a given point x.
        Args:
            x: The point at which to evaluate the linear spline.
        Returns:
            The value of the linear spline at x.
        Raises:
            ValueError: If x is outside the range of x_values.
        """
        logger.info(f"Linear spline called with x={x}")
        if x < self.x_values[0] or x > self.x_values[-1]:
            raise ValueError("x is outside the range of x_values.")
        for i in range(self.n - 1):
            if self.x_values[i] <= x <= self.x_values[i + 1]:
                result = self.y_values[i] + (self.y_values[i + 1] - self.y_values[i]) * (x - self.x_values[i]) / (self.x_values[i + 1] - self.x_values[i])
                logger.info(f"Linear spline result: S({x}) = {result} (segment{i})")
                logger.debug(f"Linear spline piecewise: \n{self.linear_spline_str()}")
                # Linear interpolation formula
                return result
        raise ValueError("x is outside the range of x_values.")

    def _compute_cubic_spline_coefficients(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Pre-compute and cache the coefficients for the cubic spline interpolation.
        """
        if self._cubic_spline_coefficients is not None:
            return self._cubic_spline_coefficients
        n=self.n
        h = [self.x_values[i + 1] - self.x_values[i] for i in range(n - 1)]
        alpha: List[float] = [0.0] * (n - 1)
        for i in range(1, n - 1):
            alpha[i] = (3 / h[i]) * (self.y_values[i + 1] - self.y_values[i]) - (3 / h[i - 1]) * (self.y_values[i] - self.y_values[i - 1])
        l: List[float] = [1.0] + [0.0] * (n - 1)
        mu: List[float] = [0.0] * (n - 1)
        z: List[float] = [0.0] * n
        for i in range(1, n - 1):
            l[i] = 2 * (self.x_values[i + 1] - self.x_values[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n - 1] = 1.0
        z[n - 1] = 0.0

        c: List[float] = [0.0] * n
        b: List[float] = [0.0] * (n - 1)
        d: List[float] = [0.0] * (n - 1)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (self.y_values[j + 1] - self.y_values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        self._cubic_spline_coefficients = (b, c, d)
        return self._cubic_spline_coefficients

    def cubic_spline(self, x: float) -> float:
        """
        Compute the cubic spline interpolation at a given point x.
        Args:
            x: The point at which to evaluate the cubic spline.
        Returns:
            The value of the cubic spline at x.
        Raises:
            ValueError: If x is outside the range of x_values.
        """

        logger.info(f"Cubic spline called with x={x}")
        if x < self.x_values[0] or x > self.x_values[-1]:
            raise ValueError("x is outside the range of x_values.")
        
        b, c, d = self._compute_cubic_spline_coefficients()
        for i in range(self.n - 1):
            if self.x_values[i] <= x <= self.x_values[i + 1]:
                dx = x - self.x_values[i]
                result = self.y_values[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
                logger.info(f"Cubic spline result: S({x}) = {result} (segment{i})")
                logger.debug(f"Cubic spline piecewise: \n{self.cubic_spline_str()}")
                return result
        raise ValueError("x is outside the range of x_values.")
    
    def __repr__(self) -> str:
        return f"Interpolation(x_values={self.x_values}, y_values={self.y_values})"

class LinearSystem:
    """
    A class to solve a system of linear equations using Gaussian elimination.
    Attributes:
        :coefficients (list): A list of lists representing the coefficient matrix.
        :constants (list): A list representing the constant terms of the equations.

    Methods:
        :gauss_elimination(): Solves the system of equations using Gaussian elimination.
        :gauss_jacobi(max_iterations, tolerance, initial_guess): Solves the system using the Gauss-Jacobi iterative method.
        :gauss_seidel(max_iterations, tolerance, initial_guess): Solves the system using the Gauss-Seidel iterative method.
    Raises:
        ValueError: If the matrix is singular or nearly singular, or if the input is invalid.
    """
    def __init__(self, coefficients: list, constants: list):
        self.coefficients = [row[:] for row in coefficients]  # Deep copy to avoid modifying the original matrix
        self.constants = constants[:]
        if len(coefficients) != len(constants):
            raise ValueError("The number of equations must match the number of constants.")
        if any(len(row) != len(coefficients) for row in coefficients):
            raise ValueError("All rows in the coefficient matrix must have the same length.")
        if not all(isinstance(c, (int, float)) for c in constants):
            raise ValueError("All constants must be numeric values.")
        if not all(isinstance(row, list) for row in coefficients):
            raise ValueError("Coefficients must be provided as a list of lists.")
        

    def gauss_elimination(self) -> List[float]:
        """ 
        Solves the system of linear equations using Gaussian elimination.
        Returns:
            list: A list containing the solution to the system of equations.
        Raises:
            ValueError: If the matrix is singular or nearly singular, or if the input is invalid.
        """

        logger.info("Gaussian elimination called.")
        n = len(self.constants)
        A = [row[:] for row in self.coefficients]  # Deep copy of the coefficient matrix
        b = self.constants[:]
        solutions: List[float] = [0.0] * n

        for k in range(n):
            max_row = k
            for i in range(k + 1, n):
                if abs(A[i][k]) > abs(A[max_row][k]):
                    max_row = i
            # Swap rows
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

            pivot = A[k][k]
            if abs(pivot) < 1e-15:
                raise ValueError("Matrix is singular or nearly singular.")
            for i in range(k + 1, n):
                factor = A[i][k] / pivot
                b[i] -= factor * b[k]
                for j in range(k, n):
                    A[i][j] -= factor * A[k][j]
        # Back substitution
        for i in range(n - 1, -1, -1):
            sum_ax = sum(A[i][j] * solutions[j] for j in range(i + 1, n))
            solutions[i] = (b[i] - sum_ax) / A[i][i]

        self.solution = solutions
        logger.info(f"Gaussian elimination result: {solutions}")
        return solutions

    def _validate_iterative(self, initial_guess: Optional[List[float]]) -> List[float]:
        """
        Common validation for iterative methods (Gauss-Jacobi and Gauss-Seidel).
        Returns the initial guess to use for the iteration, which is either the provided initial guess or a zero vector if the provided guess is invalid.
        """

        criteria_ok, problematic_rows = self.row_criteria()
        if not criteria_ok:
            logging.warning("The matrix does not satisfy the row criteria for convergence.")
            logging.warning(f"Problematic rows: {problematic_rows}")
        n = len(self.constants)

        if len(self.coefficients) != n or any(len(row) != n for row in self.coefficients):
            raise ValueError("Coefficient matrix must be square and match constants vector size.")
        
        for i in range(n):
            if self.coefficients[i][i] == 0:
                raise ValueError("Matrix is singular or nearly singular.")
        
        if initial_guess is not None:
            if len(initial_guess) != n:
                logging.warning("Initial guess size does not match number of variables. Using zero vector instead.")
                return [0.0] * n
            else:
                return list(initial_guess)
        else:
            return [0.0] * n

    def gauss_jacobi(self, max_iterations=1000, tolerance=1e-10, initial_guess: Optional[List[float]]=None) -> list: # type: ignore
        """
        Solves the system of linear equations using the Gauss-Jacobi iterative method.

        Args:
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.
            initial_guess (list): Initial guess for the solution.

        Returns:
            list: Solution vector.

        Raises:
            ValueError: If input is invalid or method does not converge.
        """
        logger.info("Gauss-Jacobi method called.")
        n = len(self.constants)
        solution = self._validate_iterative(initial_guess)

        for iteration in range(max_iterations):
            new_solution = [0.0] * n
            for i in range(n):
                s = sum(self.coefficients[i][j] * solution[j] for j in range(n) if j != i)  # Sum of non-diagonal elements
                new_solution[i] = (self.constants[i] - s) / self.coefficients[i][i]  # Update the solution for the current variable
            
            error = max(abs(new_solution[i] - solution[i]) for i in range(n))  # Check for convergence
            if error < tolerance:
                self.solution = new_solution
                logger.info(f"Gauss-Jacobi converged in {iteration+1} iterations with solution: {new_solution}")
                return new_solution
            solution = new_solution  # Update solution for next iteration
        raise ValueError("Method did not converge within the maximum number of iterations.")
 
    def row_criteria(self) -> Tuple[bool, List[int]]:
        """
        Checks if the matrix satisfies the row criteria for convergence.
        Returns:
            Tuple[bool, List[int]]: A tuple where the first element is True if the matrix satisfies the criteria,
                                    and the second element is a list of row indices that do not satisfy the criteria.
        """
        problematic_rows = []
        n = len(self.coefficients)
        for i in range(n):
            row_sum = sum(abs(self.coefficients[i][j]) for j in range(n) if j != i) # Sum of non-diagonal elements
            if abs(self.coefficients[i][i]) <= row_sum: 
                problematic_rows.append(i)
        return len(problematic_rows) == 0, problematic_rows # Return True if no problematic rows

    def gauss_seidel(self, max_iterations=1000, tolerance=1e-10, initial_guess: Optional[List[float]]=None) -> list: # type: ignore
        """
        Solves the system of linear equations using the Gauss-Seidel iterative method.
        Args:
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.
            initial_guess (list): Initial guess for the solution.
        Returns:
            list: Solution vector.
        Raises:
            ValueError: If input is invalid or method does not converge.
        """
        logger.info("Gauss-Seidel method called.")
        n = len(self.constants)
        solution = self._validate_iterative(initial_guess)

        for iteration in range(max_iterations):
            new_solution = solution[:]
            for i in range(n):
                s1 = sum(self.coefficients[i][j] * solution[j] for j in range(i))  # Sum of previously updated variables
                new_solution[i] = (self.constants[i] - s1) / self.coefficients[i][i]  # Update the solution for the current variable
            error = max(abs(new_solution[i] - solution[i]) for i in range(n))  # Check for convergence
            if error < tolerance:
                self.solution = new_solution
                logger.info(f"Gauss-Seidel converged in {iteration+1} iterations with solution: {new_solution}")
                return new_solution
            solution = new_solution
        raise ValueError("Method did not converge within the maximum number of iterations.")
    
    def __repr__(self) -> str:
        return f"LinearSystem(coefficients={self.coefficients}, constants={self.constants})"
    
    def __str__(self) -> str:
        return f"Solution vector: {self.solution}"
class Solutions:

    
    def bisection(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        print_iterations: bool = False,
        error: float = 1e-8,
        max_iter: int = 100,
    ) -> Tuple[float, int]:
        """
        Bisection method to find a root of the function f in the interval [a, b].
        Args:
            :f: The function for which to find the root.
            :a: The start of the interval.
            :b: The end of the interval.
            :print_iterations: Whether to print the details of each iteration.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have different signs.")
        logger.info(f"Bisection method called: a={a}, b={b}, error={error}, max_iter={max_iter}")
        F_a = f(a)
        if print_iterations:
            print(f"{'Iteration':^10} | {'a':^10} | {'b':^10} | {'f(a)':^10} | {'f(b)':^10} | {'Midpoint':^10} | {'f(Midpoint)':^15}")
            print("=" * 91)
        for i in range(max_iter + 1):
            midpoint = (a + b) / 2
            F_mid = f(midpoint)
            logger.debug(f"Iteration {i}: a={a:.10g}, b={b:.10g}, f(a)={F_a:.10g}, f(b)={f(b):.10g}, midpoint={midpoint:.10g}, f(midpoint)={F_mid:.10g}")
            if print_iterations:
                print(f"{i:^10} | {a:^10.6f} | {b:^10.6f} | {F_a:^10.6f} | {f(b):^10.6f} | {midpoint:^10.6f} | {F_mid:^15.6f}")
            if F_mid == 0 or abs(F_mid) < error:
                self.solution = midpoint
                logger.info(f"Bisection method converged in {i} iterations with root: {midpoint}")
                return midpoint, i
            
            if F_a * F_mid < 0:
                b = midpoint
            else:
                a = midpoint
                F_a = F_mid
        raise ValueError("Method failed after maximum iterations")

    def fixed_point(
        self,
        g: Callable[[float], float], 
        x0: float, 
        print_iterations: bool = False,
        error: float = 1e-8, 
        max_iter: int = 100
    ) -> Tuple[float, int]:
        """
        Fixed Point Iteration method to find a fixed point of the function g.
        Args:
            :g: The function for which to find the fixed point.
            :x0: The initial guess.
            :print_iterations: Whether to print the details of each iteration.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the fixed point and the number of iterations performed.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        logger.debug(f"Fixed Point Iteration method called: g={g}, x0={x0}, error={error}, max_iter={max_iter}")
        i = 0
        x_n = x0
        if print_iterations:
            print(f"{'Iteration':^10} | {'x_n':^10} | {'g(x_n)':^10} | {'Error':^10}")
            print("=" * 50)
        while i <= max_iter:
            g_xn = g(x_n)
            iter_error = abs(g_xn - x_n)
            logger.debug(f"Iteration {i}: x_n={x_n:.10g}, g(x_n)={g_xn:.10g}, error={iter_error:.10g}")
            if print_iterations:
                print(f"{i:^10} | {x_n:^10.6f} | {g(x_n):^10.6f} | {abs(g(x_n) - x_n):^10.6f}")
            x_n1 = g(x_n)
            if abs(x_n1 - x_n) < error:
                self.solution = x_n1
                logger.info(f"Fixed Point Iteration method converged in {i} iterations with fixed point: {x_n1}")
                return x_n1, i
            x_n = x_n1
            i += 1
        logger.error("Fixed Point Iteration method failed after maximum iterations")
        raise ValueError("Method failed after maximum iterations")

    def newton_raphson(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        error: float = 1e-8,
        max_iter: int = 100,
        print_iterations: bool = False
    ) -> Tuple[float, int]:
        """
        Newton-Raphson method to find a root of the function f.
        Args:
            :f: The function for which to find the root.
            :df: The derivative of the function f.
            :x0: The initial guess.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
            :print_iterations: Whether to print the details of each iteration.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        logger.info(f"Newton-Raphson method called: f={f}, df={df}, x0={x0}, error={error}, max_iter={max_iter}")
        i = 0
        x_n = x0
        if print_iterations:
            print(f"{'Iteration':^10} | {'x_n':^10} | {'f(x_n)':^10} | {'Error':^10}")
            print("=" * 50)
        while i <= max_iter:
            logger.debug(f"Iteration {i}: x_n={x_n:.10g}, f(x_n)={f(x_n):.10g}, error={abs(f(x_n)):.10g}")
            if print_iterations:
                print(f"{i:^10} | {x_n:^10.6f} | {f(x_n):^10.6f} | {abs(f(x_n)):^10.6f}")
            df_xn = df(x_n)
            if df_xn == 0:
                raise ValueError("Derivative is zero. No solution found.")
            x_n1 = x_n - f(x_n) / df_xn
            if abs(x_n1 - x_n) < error:
                self.solution = x_n1
                logger.info(f"Newton-Raphson method converged in {i} iterations with root: {x_n1}")
                return x_n1, i
            x_n = x_n1
            i += 1
        logger.error("Newton-Raphson method failed after maximum iterations")
        raise ValueError("Method failed after maximum iterations")

    def secant(
        self,
        f: Callable[[float], float], 
        x0: float, 
        x1: float, 
        error: float = 1e-8, 
        max_iter: int = 100,
        print_iterations: bool = False
    ) -> Tuple[float, int]:
        """
        Secant method to find a root of the function f.
        Args:
            :f: The function for which to find the root.
            :x0: The first initial guess.
            :x1: The second initial guess.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
            :print_iterations: Whether to print the details of each iteration.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        logger.info(f"Secant method called: f={f}, x0={x0}, x1={x1}, error={error}, max_iter={max_iter}")
        i = 0
        if print_iterations:
            print(f"{'Iteration':^10} | {'x0':^10} | {'x1':^10} | {'x2':^10}")
            print("=" * 50)
        while i <= max_iter:
            f_x0 = f(x0)
            f_x1 = f(x1)
            if f_x1 - f_x0 == 0:
                raise ValueError("Division by zero. No solution found.")
            x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            logger.debug(f'Secant iter {i}: x0={x0:.10g}, x1={x1:.10g}, f(x0)={f_x0:.10g}, f(x1)={f_x1:.10g}, x2={x2:.10g}')
            if print_iterations:
                print(f"{i:^10} | {x0:^10.6f} | {x1:^10.6f} | {x2:^10.6f}")
            if abs(x2 - x1) < error:
                self.solution = x2
                logger.info(f"Secant method converged in {i} iterations with root: {x2}")
                return x2, i
            x0, x1 = x1, x2
            i += 1
        logger.error("Secant method failed after maximum iterations")
        raise ValueError("Method failed after maximum iterations")

    def regula_falsi(
        self,
        f: Callable[[float], float], 
        a: float, 
        b: float, 
        error: float = 1e-8, 
        max_iter: int = 100,
        print_iterations: bool = False
    ) -> Tuple[float, int]:
        """
        Regula Falsi method to find a root of the function f in the interval [a, b].
        Args:
            :f: The function for which to find the root.
            :a: The start of the interval.
            :b: The end of the interval.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
            :print_iterations: Whether to print the details of each iteration.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have different signs.")
        logger.info(f"Regula Falsi method called: a={a}, b={b}, error={error}, max_iter={max_iter}")
        F_a = f(a)
        F_b = f(b)
        if print_iterations:
            print(f"{'Iteration':^10} | {'a':^10} | {'b':^10} | {'x':^10}")
            print("=" * 50)
        for i in range(max_iter + 1):
            x = (a * F_b - b * F_a) / (F_b - F_a)
            F_x = f(x)
            logger.debug(f"Regula Falsi iter {i}: a={a:.10g}, b={b:.10g}, x={x:.10g}, f(x)={F_x:.10g}")
            if print_iterations:
                print(f"{i:^10} | {a:^10.6f} | {b:^10.6f} | {x:^10.6f}")
            if F_x == 0 or abs(F_x) < error:
                self.solution = x
                logger.info(f"Regula Falsi method converged in {i} iterations with root: {x}")
                return x, i

            if F_a * F_x > 0:
                a, F_a = x, F_x
            else:
                b, F_b = x, F_x
            i += 1
        logger.error("Regula Falsi method failed after maximum iterations")
        raise ValueError("Method failed after maximum iterations")

    def muller(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        error: float = 1e-8,
        max_iter: int = 100,
        print_iterations: bool = False
    ):
        """
        Muller method to find a root of the function f.
        Args:
            :f: The function for which to find the root.
            :x0: The first initial guess.
            :x1: The second initial guess.
            :x2: The third initial guess.
            :error: The acceptable error margin.
            :max_iter: The maximum number of iterations to perform.
            :print_iterations: Whether to print the details of each iteration.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        logger.info(f"Muller method called: x0={x0}, x1={x1}, x2={x2}, error={error}, max_iter={max_iter}")
        i = 0
        if print_iterations:
            print(f"{'Iteration':^10} | {'x0':^10} | {'x1':^10} | {'x2':^10} | {'x3':^10}")
            print("=" * 60)
        while i <= max_iter:
            f_x0 = f(x0)
            f_x1 = f(x1)
            f_x2 = f(x2)

            h0 = x1 - x0
            h1 = x2 - x1
            if h0 == 0 or h1 == 0:
                raise ValueError("Division by zero. No solution found.")
            delta0 = (f_x1 - f_x0) / h0
            delta1 = (f_x2 - f_x1) / h1
            a = (delta1 - delta0) / (h1 + h0)
            b = a * h1 + delta1
            c = f_x2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                raise ValueError("Complex root encountered. No solution found.")
            sqrt_discriminant = discriminant**0.5

            if abs(b + sqrt_discriminant) > abs(b - sqrt_discriminant):
                denominator = b + sqrt_discriminant
            else:
                denominator = b - sqrt_discriminant
            if denominator == 0:
                raise ValueError("Division by zero. No solution found.")
            x3 = x2 - (2 * c) / denominator
            logger.debug(f"Muller iter {i}: x0={x0:10.6f}, x1={x1:10.6f}, x2={x2:10.6f}, f(x0)={f_x0:10.6f}, f(x1)={f_x1:10.6f}, f(x2)={f_x2:10.6f}, x3={x3:10.6f}")
            if print_iterations:
                print(f"{i:^10} | {x0:^10.6f} | {x1:^10.6f} | {x2:^10.6f} | {x3:^10.6f}")
            if abs(x3 - x2) < error:
                self.solution = x3
                logger.info(f"Muller method converged in {i} iterations with root: {x3}")
                return x3, i

            x0, x1, x2 = x1, x2, x3
            i += 1
        logger.error("Muller method failed after maximum iterations")
        raise ValueError("Method failed after maximum iterations")
    
    def __repr__(self) -> str:
        return f"Solutions(solution={getattr(self, 'solution', None)})"
    
    def __str__(self) -> str:
        return f"Root: {getattr(self, 'solution', None)}"
    


# # -- Testing all methods with a simple example --
# if __name__ == "__main__":
#     sol = Solutions()
#     root, iterations = sol.bisection(lambda x: x**2 - 2, 0, 2, print_iterations=True)
#     print(f"Bisection method found root: {root} in {iterations} iterations")

#     root, iterations = sol.fixed_point(lambda x: (x + 2 / x) / 2, 1.0, print_iterations=True)
#     print(f"Fixed Point method found root: {root} in {iterations} iterations")

#     root, iterations = sol.newton_raphson(lambda x: x**2 - 2, lambda x: 2*x, 1.0, print_iterations=True)
#     print(f"Newton-Raphson method found root: {root} in {iterations} iterations")

#     root, iterations = sol.secant(lambda x: x**2 - 2, 1.0, 2.0, print_iterations=True)
#     print(f"Secant method found root: {root} in {iterations} iterations")

#     root, iterations = sol.regula_falsi(lambda x: x**2 - 2, 0, 2, print_iterations=True)
#     print(f"Regula Falsi method found root: {root} in {iterations} iterations")

#     root, iterations = sol.muller(lambda x: x**2 - 2, 0.0, 1.0, 2.0, print_iterations=True)
#     print(f"Muller method found root: {root} in {iterations} iterations")

#     interp = Interpolation([0, 1, 2, 3, 4, 5], [1, 2, 0, 5, 7, -2])
#     x = 1.5
    
#     print(f"Lagrange interpolation at x={x}: {interp.lagrange(x)}")
#     print(interp.cubic_spline_str())

#     print(f"Newton interpolation at x={x}: {interp.newton(x)}")
#     print(interp.newton_polynomial_str())

#     print(f"Linear spline interpolation at x={x}: {interp.linear_spline(x)}")
#     print(interp.linear_spline_str())

#     print(f"Cubic spline interpolation at x={x}: {interp.cubic_spline(x)}")
#     print(interp.cubic_spline_str())

#     coefficients = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
#     constants = [6, 25, -11]
#     system = LinearSystem(coefficients, constants)
#     print(f"Gaussian elimination solution: {system.gauss_elimination()}")
#     print(f"Gauss-Jacobi solution: {system.gauss_jacobi()}")
#     print(f"Gauss-Seidel solution: {system.gauss_seidel()}")