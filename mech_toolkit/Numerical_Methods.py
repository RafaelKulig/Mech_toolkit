from typing import List, Tuple, Callable
class Interpolation:
    
    def __init__(self, x_values: List[float], y_values: List[float]):
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length.")    # Ensure both lists have the same length
        if len(set(x_values)) != len(x_values): # Check for distinct x_values
            raise ValueError("x_values must be distinct for interpolation.")
        self.x_values = x_values
        self.y_values = y_values
        self.n = len(x_values)

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
        result = 0.0
        for i in range(self.n):
            term = self.y_values[i]
            for j in range(self.n):
                if j != i:
                    term *= (x - self.x_values[j]) / (self.x_values[i] - self.x_values[j])
            result += term
        return result
    
    def _divided_differences(self) -> list:
        """Compute the divided differences table."""
        n = self.n
        coef = [[0 for _ in range(n)] for _ in range(n)]
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
        coef = self._divided_differences()
        result = coef[0][0]
        term = 1.0
        for i in range(1, self.n):
            term *= (x - self.x_values[i - 1])
            result += coef[0][i] * term
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
        if x < self.x_values[0] or x > self.x_values[-1]:
            raise ValueError("x is outside the range of x_values.")
        for i in range(self.n - 1):
            if self.x_values[i] <= x <= self.x_values[i + 1]:
                # Linear interpolation formula
                return self.y_values[i] + (self.y_values[i + 1] - self.y_values[i]) * (x - self.x_values[i]) / (self.x_values[i + 1] - self.x_values[i])
        raise ValueError("x is outside the range of x_values.")

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
        if x < self.x_values[0] or x > self.x_values[-1]:
            raise ValueError("x is outside the range of x_values.")
        n = self.n
        h = [self.x_values[i + 1] - self.x_values[i] for i in range(n - 1)]
        alpha = [0] * (n - 1)
        for i in range(1, n - 1):
            alpha[i] = (3 / h[i]) * (self.y_values[i + 1] - self.y_values[i]) - (3 / h[i - 1]) * (self.y_values[i] - self.y_values[i - 1])
        l = [1] + [0] * (n - 1)
        mu = [0] * (n - 1)
        z = [0] * n
        for i in range(1, n - 1):
            l[i] = 2 * (self.x_values[i + 1] - self.x_values[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n - 1] = 1
        z[n - 1] = 0
        c = [0] * n
        b = [0] * (n - 1)
        d = [0] * (n - 1)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (self.y_values[j + 1] - self.y_values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        for i in range(n - 1):
            if self.x_values[i] <= x <= self.x_values[i + 1]:
                dx = x - self.x_values[i]
                return self.y_values[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        raise ValueError("x is outside the range of x_values.")

class LinearSystem:
    """
    A class to solve a system of linear equations using Gaussian elimination.
    Attributes:
        coefficients (list): A list of lists representing the coefficient matrix.
        constants (list): A list representing the constant terms of the equations.

    Methods:
        gauss_elimination(): Solves the system of equations using Gaussian elimination.
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
        

    def gauss_elimination(self) -> list:
        """ 
        Solves the system of linear equations using Gaussian elimination.
        Returns:
            list: A list containing the solution to the system of equations.
        Raises:
            ValueError: If the matrix is singular or nearly singular, or if the input is invalid.
        """

        n = len(self.constants)
        solution = [0] * n  # Initialize the solution vector with zeros

        # Transform the matrix to upper triangular form
        for k in range(n): # Iterate over each column
            pivot=self.coefficients[k][k] # Get the pivot element
            if pivot == 0:
                raise ValueError("Matrix is singular or nearly singular.")
            for i in range(k + 1, n): # Iterate over each row below the current row
                factor = self.coefficients[i][k] / pivot # Calculate the factor to eliminate the variable
                self.constants[i] -= factor * self.constants[k] # Adjust the constant term accordingly
                for j in range(k, n): # Iterate over each column to the right of the current column
                    self.coefficients[i][j] -= factor * self.coefficients[k][j] # Eliminate the variable in the current row
        
        # Back substitution to find the solution
        for i in range(n - 1, -1, -1): # Back substitution
            s=sum(self.coefficients[i][j] * solution[j] for j in range(i + 1, n)) # Calculate the sum of known variables
            solution[i] = (self.constants[i] - s) / self.coefficients[i][i] # Solve for the current variable
        return solution
    def gauss_jacobi(self, max_iterations=1000, tolerance=1e-10, initial_guess: List[float]=None) -> list: # type: ignore
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
        criteria_ok, problematic_rows = self.row_criteria()
        if not criteria_ok:
            print("Warning: The matrix does not satisfy the row criteria for convergence.")
            print(f"Problematic rows: {problematic_rows}")

        n = len(self.constants)

        # Validate input dimensions
        if len(self.coefficients) != n or any(len(row) != n for row in self.coefficients):
            raise ValueError("Coefficient matrix must be square and match constants vector size.")

        # Check for zero diagonal elements
        for i in range(n):
            if self.coefficients[i][i] == 0:
                raise ValueError("Matrix is singular or nearly singular.")
        if initial_guess:
            if len(initial_guess) != n:
                print("Warning: Initial guess size does not match number of variables. Using zero vector instead.")
                solution = [0.0] * n
            else:
                solution = list(initial_guess)
        else:
            solution = [0.0] * n

        for iteration in range(max_iterations):
            new_solution = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):  
                    if j!=i:
                        s+=self.coefficients[i][j]*solution[j]  # Sum of known variables
                new_solution[i] = (self.constants[i] - s) / self.coefficients[i][i] # Update the solution for the current variable

            # Check for convergence
            error = max(abs(new_solution[i] - solution[i]) for i in range(n))   
            if error < tolerance:
                return new_solution

            solution = new_solution
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

    def gauss_seidel(self, max_iterations=1000, tolerance=1e-10, initial_guess: List[float]=None) -> list: # type: ignore
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
        criteria_ok, problematic_rows = self.row_criteria()
        if not criteria_ok:
            print("Warning: The matrix does not satisfy the row criteria for convergence.")
            print(f"Problematic rows: {problematic_rows}")

        n = len(self.constants)

        # Validate input dimensions
        if len(self.coefficients) != n or any(len(row) != n for row in self.coefficients):
            raise ValueError("Coefficient matrix must be square and match constants vector size.")

        # Check for zero diagonal elements
        for i in range(n):
            if self.coefficients[i][i] == 0:
                raise ValueError("Matrix is singular or nearly singular.")
        if initial_guess:
            if len(initial_guess) != n:
                print("Warning: Initial guess size does not match number of variables. Using zero vector instead.")
                solution = [0.0] * n
            else:
                solution = list(initial_guess)
        else:
            solution = [0.0] * n

        for iteration in range(max_iterations):
            new_solution = solution[:]
            for i in range(n):
                s = 0.0
                for j in range(n):  
                    if j!=i:
                        s+=self.coefficients[i][j]*new_solution[j]  # Use the most recent values
                new_solution[i] = (self.constants[i] - s) / self.coefficients[i][i] # Update the solution for the current variable
            # Check for convergence
            error = max(abs(new_solution[i] - solution[i]) for i in range(n))   
            if error < tolerance:
                return new_solution
            solution = new_solution
        raise ValueError("Method did not converge within the maximum number of iterations.")

class Solutions:

    @staticmethod
    def bisection(
        f: Callable[[float], float],
        a: float,
        b: float,
        error: float,
        max_iter: int,
    ):
        """
        Bisection method to find a root of the function f in the interval [a, b].
        Args:
            f: The function for which to find the root.
            a: The start of the interval.
            b: The end of the interval.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have different signs.")
        F_a = f(a)
        i = 0
        while i <= max_iter:
            if (F_x := f(x := (a + b) / 2)) == 0 or (b - a) / 2 < error:
                return x, i

            if F_a * F_x > 0:
                a, F_a = x, F_x
            else:
                b = x
            i += 1
        raise ValueError("Method failed after maximum iterations")

    @staticmethod
    def fixed_point(
        g: Callable[[float], float], 
        x0: float, 
        error: float, 
        max_iter: int
    ):
        """
        Fixed Point Iteration method to find a fixed point of the function g.
        Args:
            g: The function for which to find the fixed point.
            x0: The initial guess.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the fixed point and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        i = 0
        x_n = x0
        while i <= max_iter:
            x_n1 = g(x_n)
            if abs(x_n1 - x_n) < error:
                return x_n1, i
            x_n = x_n1
            i += 1
        raise ValueError("Method failed after maximum iterations")

    @staticmethod
    def newton_raphson(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        error: float,
        max_iter: int,
    ):
        """
        Newton-Raphson method to find a root of the function f.
        Args:
            f: The function for which to find the root.
            df: The derivative of the function f.
            x0: The initial guess.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        i = 0
        x_n = x0
        while i <= max_iter:
            df_xn = df(x_n)
            if df_xn == 0:
                raise ValueError("Derivative is zero. No solution found.")
            x_n1 = x_n - f(x_n) / df_xn
            if abs(x_n1 - x_n) < error:
                return x_n1, i
            x_n = x_n1
            i += 1
        raise ValueError("Method failed after maximum iterations")

    @staticmethod
    def secant(
        f: Callable[[float], float], 
        x0: float, 
        x1: float, 
        error: float, 
        max_iter: int
    ):
        """
        Secant method to find a root of the function f.
        Args:
            f: The function for which to find the root.
            x0: The first initial guess.
            x1: The second initial guess.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        i = 0
        while i <= max_iter:
            f_x0 = f(x0)
            f_x1 = f(x1)
            if f_x1 - f_x0 == 0:
                raise ValueError("Division by zero. No solution found.")
            x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            if abs(x2 - x1) < error:
                return x2, i
            x0, x1 = x1, x2
            i += 1
        raise ValueError("Method failed after maximum iterations")

    @staticmethod
    def regula_falsi(
        f: Callable[[float], float], 
        a: float, 
        b: float, 
        error: float, 
        max_iter: int
    ):
        """
        Regula Falsi method to find a root of the function f in the interval [a, b].
        Args:
            f: The function for which to find the root.
            a: The start of the interval.
            b: The end of the interval.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have different signs.")
        F_a = f(a)
        F_b = f(b)
        i = 0
        while i <= max_iter:
            if (F_x := f(x := (a * F_b - b * F_a) / (F_b - F_a))) == 0 or abs(
                F_x
            ) < error:
                return x, i

            if F_a * F_x > 0:
                a, F_a = x, F_x
            else:
                b, F_b = x, F_x
            i += 1
        raise ValueError("Method failed after maximum iterations")

    @staticmethod
    def muller(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        error: float,
        max_iter: int,
    ):
        """
        Muller method to find a root of the function f.
        Args:
            f: The function for which to find the root.
            x0: The first initial guess.
            x1: The second initial guess.
            x2: The third initial guess.
            error: The acceptable error margin.
            max_iter: The maximum number of iterations to perform.
        Returns:
            A tuple containing the root and the number of iterations performed.
        Raises:
            ValueError: If the method fails to converge within the maximum number of iterations.
        """
        if error <= 0 or max_iter <= 0:
            raise ValueError("Error and max_iter must be positive values.")
        i = 0
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

            if abs(x3 - x2) < error:
                return x3, i

            x0, x1, x2 = x1, x2, x3
            i += 1
        raise ValueError("Method failed after maximum iterations")