import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from types import SimpleNamespace


class MarketEquilibrium:
    @staticmethod
    def market_clearing_error(p1, omega_A1, omega_A2):
        p2 = 1  # Numeraire
        alpha = 1/3
        beta = 2/3
        omega_B1 = 1 - omega_A1
        omega_B2 = 1 - omega_A2

        # Demand functions
        xA_star_1 = alpha * (p1 * omega_A1 + p2 * omega_A2) / p1
        xB_star_1 = beta * (p1 * omega_B1 + p2 * omega_B2) / p1
        xA_star_2 = (1 - alpha) * (p1 * omega_A1 + p2 * omega_A2) / p2
        xB_star_2 = (1 - beta) * (p1 * omega_B1 + p2 * omega_B2) / p2

        # Market clearing errors
        error1 = xA_star_1 + xB_star_1 - (omega_A1 + omega_B1)
        error2 = xA_star_2 + xB_star_2 - (omega_A2 + omega_B2)
        return abs(error1) + abs(error2)

    @staticmethod
    def find_equilibrium(omega_A1, omega_A2):
        result = minimize(MarketEquilibrium.market_clearing_error, x0=1, args=(omega_A1, omega_A2), bounds=[(0.01, 10)])
        return result.x[0]


class EconomicModel:
    # Constants
    alpha = 1/3
    beta = 2/3
    N = 75
    omega_A1 = 0.8
    omega_A2 = 0.3
    omega_B1 = 1 - omega_A1
    omega_B2 = 1 - omega_A2
    p2 = 1  # Numeraire

    @staticmethod
    def utility_A(x1, x2):
        return x1**EconomicModel.alpha * x2**(1 - EconomicModel.alpha)

    @staticmethod
    def utility_B(x1, x2):
        return x1**EconomicModel.beta * x2**(1 - EconomicModel.beta)

    @staticmethod
    def initial_utilities():
        uA_initial = EconomicModel.utility_A(EconomicModel.omega_A1, EconomicModel.omega_A2)
        uB_initial = EconomicModel.utility_B(EconomicModel.omega_B1, EconomicModel.omega_B2)
        return uA_initial, uB_initial

    @staticmethod
    def calculate_pareto_improvements(N=75):
        x_A1_range = np.linspace(0, 1, N)
        x_A2_range = np.linspace(0, 1, N)
        X_A1, X_A2 = np.meshgrid(x_A1_range, x_A2_range)

        # Calculate utilities at each point
        UA = EconomicModel.utility_A(X_A1, X_A2)
        UB = EconomicModel.utility_B(1 - X_A1, 1 - X_A2)

        # Initial utilities
        uA_initial, uB_initial = EconomicModel.initial_utilities()

        # Mask to find combinations where both A and B are at least as well off
        pareto_improvements = (UA >= uA_initial) & (UB >= uB_initial)
        
        return X_A1, X_A2, pareto_improvements


class ExchangeEconomy1:
    def __init__(self, alpha, beta, w1A, w2A):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A

    def demand_A(self, p1):
        # Demand function for consumer A
        income_A = p1 * self.w1A + self.w2A
        return (self.alpha * income_A / p1, (1 - self.alpha) * income_A)

    def demand_B(self, p1):
        # Demand function for consumer B
        income_B = p1 * self.w1B + self.w2B
        return (self.beta * income_B / p1, (1 - self.beta) * income_B)

    def compute_errors(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        # Excess demand for good 1 and good 2
        eps1 = x1A + x1B - 1
        eps2 = x2A + x2B - 1
        return eps1, eps2

    def excess_demand_good1(self, p1):
        eps1, _ = self.compute_errors(p1)
        return eps1

    def find_market_clearing_price(self, bracket=[0.5, 2.5]):
        result = root_scalar(self.excess_demand_good1, bracket=bracket, method='brentq')
        if result.converged:
            return result.root
        else:
            raise ValueError('Could not find a market-clearing price.')

    def compute_market_clearing_errors(self, p1_values):
        errors = [self.compute_errors(p1) for p1 in p1_values]
        return zip(*[(p1, eps[0], eps[1]) for p1, eps in zip(p1_values, errors)])


class MarketClearing:
    # Constants for the two consumers and initial endowments
    alpha = 1/3
    beta = 2/3
    omega_A1 = 0.8
    omega_A2 = 0.3
    omega_B1 = 1 - omega_A1
    omega_B2 = 1 - omega_A2
    p2 = 1  # Numeraire

    @staticmethod
    def excess_demand_x1(p1):
        # Demand for good 1 by consumers A and B
        demand_A1 = MarketClearing.alpha * (p1 * MarketClearing.omega_A1 + MarketClearing.p2 * MarketClearing.omega_A2) / p1
        demand_B1 = MarketClearing.beta * (p1 * MarketClearing.omega_B1 + MarketClearing.p2 * MarketClearing.omega_B2) / p1
        # Total excess demand for good 1
        return demand_A1 + demand_B1 - (MarketClearing.omega_A1 + MarketClearing.omega_B1)

    @staticmethod
    def excess_demand_x2(p1):
        # Demand for good 2 by consumers A and B
        demand_A2 = (1 - MarketClearing.alpha) * (p1 * MarketClearing.omega_A1 + MarketClearing.p2 * MarketClearing.omega_A2) / MarketClearing.p2
        demand_B2 = (1 - MarketClearing.beta) * (p1 * MarketClearing.omega_B1 + MarketClearing.p2 * MarketClearing.omega_B2) / MarketClearing.p2
        # Total excess demand for good 2
        return demand_A2 + demand_B2 - (MarketClearing.omega_A2 + MarketClearing.omega_B2)

    @staticmethod
    def total_excess_demand(p1):
        ed1 = MarketClearing.excess_demand_x1(p1)
        ed2 = MarketClearing.excess_demand_x2(p1)
        return np.array([ed1, ed2])

    @staticmethod
    def find_market_clearing_price(p1_initial=1.0, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
        p1 = p1_initial
        for iteration in range(max_iterations):
            excess_demand = MarketClearing.total_excess_demand(p1)
            # Check if the excess demand is within the desired tolerance level
            if np.abs(excess_demand[0]) < tolerance and np.abs(excess_demand[1]) < tolerance:
                print(f"Converged after {iteration} iterations")
                return p1
            # Update price using excess demand sign (proportional to excess demand to guide correction)
            p1 += learning_rate * excess_demand[0]
        return p1


class ExchangeEconomy:
    def __init__(self, alpha, beta, w1A, w2A):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A

    def utility_A(self, x1A, x2A):
        # Ensure x1A and x2A are positive before calculation
        if x1A <= 0 or x2A <= 0:
            return -np.inf  # Return a very low utility if any demand is non-positive
        return x1A**self.alpha * x2A**(1 - self.alpha)

    def demand_A(self, p1):
        income_A = p1 * self.w1A + self.w2A  # Assuming p2 (price of good 2) is normalized to 1
        return self.alpha * income_A / p1, (1 - self.alpha) * income_A

    def demand_B(self, p1):
        income_B = p1 * self.w1B + self.w2B  # Assuming p2 (price of good 2) is normalized to 1
        return self.beta * income_B / p1, (1 - self.beta) * income_B

def find_optimal_allocation(economy, P1):
    max_utility = -np.inf
    optimal_price = None
    optimal_allocation = None

    for p1 in P1:
        demandB = economy.demand_B(p1)
        x1A = 1 - demandB[0]
        x2A = 1 - demandB[1]
        utility = economy.utility_A(x1A, x2A)

        if utility > max_utility:
            max_utility = utility
            optimal_price = p1
            optimal_allocation = (x1A, x2A)

    return optimal_price, optimal_allocation

def optimize_all_positive(economy):
    def objective(p1):
        if p1 <= 0:
            return np.inf  # ensures p1 is positive
        demandB = economy.demand_B(p1)
        x1A = 1 - demandB[0]
        x2A = 1 - demandB[1]
        return -economy.utility_A(x1A, x2A)  # Minimize the negative utility to find maximum

    result = minimize_scalar(objective, bounds=(0.01, 10), method='bounded')
    optimal_p1 = result.x
    optimal_allocation = (1 - economy.demand_B(optimal_p1)[0],
                          1 - economy.demand_B(optimal_p1)[1])
    return optimal_p1, optimal_allocation


class ExchangeEconomyClass:
    def __init__(self, alpha, beta, w1A, w2A):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        # Assuming the total endowment of both goods is normalized to 1
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A
        # Initialize utilities at the initial endowment
        self.utility_A_initial = self.utility_A(w1A, w2A)
        self.utility_B_initial = self.utility_B(self.w1B, self.w2B)

    def utility_A(self, x1A, x2A):
        # Define utility function for consumer A
        return x1A ** self.alpha * x2A ** (1 - self.alpha)

    def utility_B(self, x1B, x2B):
        # Define utility function for consumer B
        return x1B ** self.beta * x2B ** (1 - self.beta)

    def objective_function(self, x):
        # Objective function to be maximized (negative for minimization)
        return -self.utility_A(x[0], x[1])

    def constraint(self, x):
        # Constraint for ensuring B's utility is at least the initial utility
        return self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B_initial

    def optimize_allocation(self):
        # Define constraints and bounds
        constraints = ({'type': 'ineq', 'fun': self.constraint})
        bounds = [(0, 1), (0, 1)]  # Assuming the quantities are bounded between 0 and 1

        # Perform the optimization
        initial_guess = [self.w1A, self.w2A]
        result = minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_allocation_A = result.x
            optimal_utility_A = -result.fun
            return optimal_allocation_A, optimal_utility_A
        else:
            raise ValueError("Optimization failed.")


import numpy as np

class ExchangeEconomy2:
    def __init__(self, alpha, beta, w1A, w2A):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A
        self.utility_A_initial = self.utility_A(w1A, w2A)
        self.utility_B_initial = self.utility_B(self.w1B, self.w2B)

    def utility_A(self, x1A, x2A):
        return max(x1A, 0) ** self.alpha * max(x2A, 0) ** (1 - self.alpha)

    def utility_B(self, x1B, x2B):
        return max(x1B, 0) ** self.beta * max(x2B, 0) ** (1 - self.beta)

    def is_feasible(self, x1A, x2A):
        x1B = 1 - x1A
        x2B = 1 - x2A
        return self.utility_A(x1A, x2A) >= self.utility_A_initial and self.utility_B(x1B, x2B) >= self.utility_B_initial

    def find_optimal_allocation(self):
        N = 100  # Resolution for allocations
        x1A_range = np.linspace(0, 1, N)
        x2A_range = np.linspace(0, 1, N)
        max_utility_A = -np.inf
        optimal_allocation_A = None

        for x1A in x1A_range:
            for x2A in x2A_range:
                if self.is_feasible(x1A, x2A):
                    utility_A = self.utility_A(x1A, x2A)
                    if utility_A > max_utility_A:
                        max_utility_A = utility_A
                        optimal_allocation_A = (x1A, x2A)

        return optimal_allocation_A, max_utility_A


class ExchangeEconomy3:
    def __init__(self, alpha, beta, w1A, w2A, w1B, w2B):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = w1B
        self.w2B = w2B

    def utility_A(self, x1A, x2A):
        # Define the utility function for consumer A
        return x1A ** self.alpha * x2A ** (1 - self.alpha)

    def utility_B(self, x1B, x2B):
        # Define the utility function for consumer B
        return x1B ** self.beta * x2B ** (1 - self.beta)

    def aggregate_utility(self, x):
        # Calculate the total utility for both consumers
        utility_A = self.utility_A(x[0], x[1])
        utility_B = self.utility_B(x[2], x[3])
        return -(utility_A + utility_B)  # Negative for maximization

    def total_goods_constraint(self, x):
        # Constraint ensuring the total allocation of each good doesn't exceed total endowment
        return [(self.w1A + self.w1B - (x[0] + x[2])), (self.w2A + self.w2B - (x[1] + x[3]))]

    def optimize_allocation(self):
        # Set up the optimization problem
        initial_guess = [self.w1A, self.w2A, self.w1B, self.w2B]
        constraints = [{'type': 'eq', 'fun': lambda x: self.total_goods_constraint(x)[0]},
                       {'type': 'eq', 'fun': lambda x: self.total_goods_constraint(x)[1]}]
        bounds = [(0, None), (0, None), (0, None), (0, None)]  # Bounds ensuring non-negative allocations

        # Perform the optimization
        result = minimize(self.aggregate_utility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_allocation = result.x
            optimal_aggregate_utility = -result.fun
            return optimal_allocation, optimal_aggregate_utility
        else:
            raise ValueError("Optimization failed.")
