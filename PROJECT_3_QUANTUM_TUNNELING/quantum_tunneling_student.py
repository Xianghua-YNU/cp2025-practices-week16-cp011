"""Module: Quantum Tunneling Solution
File: quantum_tunneling_solution.py

This module implements a numerical solution for the 1D time-dependent Schrödinger equation
to simulate quantum tunneling through a potential barrier using the Crank-Nicolson method.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class QuantumTunnelingSolver:
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """
        Initialize the quantum tunneling solver with simulation parameters.
        
        Parameters:
        - Nx: Number of spatial grid points (default: 220)
        - Nt: Number of time steps (default: 300)
        - x0: Initial position of the wave packet center (default: 40)
        - k0: Initial wave number (default: 0.5)
        - d: Width parameter of the Gaussian wave packet (default: 10)
        - barrier_width: Width of the potential barrier (default: 3)
        - barrier_height: Height of the potential barrier (default: 1.0)
        """
        self.Nx = Nx  # Number of spatial points
        self.Nt = Nt  # Number of time steps
        self.x0 = x0  # Initial wave packet center position
        self.k0 = k0  # Initial wave number
        self.d = d   # Gaussian wave packet width parameter
        self.barrier_width = barrier_width  # Width of potential barrier
        self.barrier_height = barrier_height  # Height of potential barrier
        self.x = np.arange(self.Nx)  # Spatial grid
        self.V = self.setup_potential()  # Potential energy array
        self.C = np.zeros((self.Nx, self.Nt), complex)  # Wave function coefficient matrix
        self.B = np.zeros((self.Nx, self.Nt), complex)  # Wave function coefficient matrix

    def wavefun(self, x):
        """Calculate the initial Gaussian wave packet function at position x.
        
        Returns:
        Complex value representing the wave function at position x.
        """
        return np.exp(self.k0*1j*x)*np.exp(-(x-self.x0)**2*np.log10(2)/self.d**2)

    def setup_potential(self):
        """Setup the rectangular potential barrier.
        
        Returns:
        Numpy array representing the potential energy at each spatial point.
        """
        self.V = np.zeros(self.Nx)  # Initialize potential as zero everywhere
        # Set potential barrier in the center
        self.V[self.Nx//2:self.Nx//2+self.barrier_width] = self.barrier_height
        return self.V

    def build_coefficient_matrix(self):
        """Build the coefficient matrix for the Crank-Nicolson scheme.
        
        Returns:
        Tridiagonal matrix representing the discretized Hamiltonian.
        """
        # Create tridiagonal matrix with main diagonal and upper/lower diagonals
        A = np.diag(-2+2j-self.V) + np.diag(np.ones(self.Nx-1),1) + np.diag(np.ones(self.Nx-1),-1)
        return A

    def solve_schrodinger(self):
        """Solve the 1D time-dependent Schrödinger equation using Crank-Nicolson method.
        
        Returns:
        Tuple containing spatial grid, potential, and wave function coefficients.
        """
        A = self.build_coefficient_matrix()  # Get coefficient matrix
        
        # Initialize wave function at t=0
        self.B[:,0] = self.wavefun(self.x)
        
        # Time evolution using Crank-Nicolson method
        for t in range(self.Nt-1):
            # Solve matrix equation for next time step
            self.C[:,t+1] = 4j*np.linalg.solve(A, self.B[:,t])
            # Update wave function
            self.B[:,t+1] = self.C[:,t+1] - self.B[:,t]
        
        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """Calculate transmission and reflection coefficients.
        
        Returns:
        Tuple containing transmission and reflection probabilities.
        """
        barrier_position = len(self.x)//2  # Center position of barrier
        
        # Calculate probability beyond the barrier (transmitted)
        transmitted_prob = np.sum(np.abs(self.B[barrier_position+self.barrier_width:, -1])**2)
        # Calculate probability before the barrier (reflected)
        reflected_prob = np.sum(np.abs(self.B[:barrier_position, -1])**2)
        # Total probability for normalization
        total_prob = np.sum(np.abs(self.B[:, -1])**2)
        
        return transmitted_prob/total_prob, reflected_prob/total_prob

    def plot_evolution(self, time_indices=None):
        """Plot wave function evolution at specific time steps.
        
        Parameters:
        - time_indices: List of time indices to plot (default: 5 evenly spaced times)
        """
        if time_indices is None:
            Nt = self.B.shape[1]
            # Default to 5 evenly spaced time points
            time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        # Add overall title with barrier parameters
        fig.suptitle(f'Quantum Tunneling Evolution - Barrier Width: {self.barrier_width}, Barrier Height: {self.barrier_height}', 
                     fontsize=14, fontweight='bold')
        
        # Plot each specified time step
        for i, t_idx in enumerate(time_indices):
            if i < len(axes):
                ax = axes[i]
                
                # Plot probability density (|ψ|²)
                prob_density = np.abs(self.B[:, t_idx])**2
                ax.plot(self.x, prob_density, 'b-', linewidth=2, 
                       label=f'|ψ|² at t={t_idx}')
                
                # Plot potential barrier
                ax.plot(self.x, self.V, 'k-', linewidth=2, 
                       label=f'Barrier (Width={self.barrier_width}, Height={self.barrier_height})')
                
                ax.set_xlabel('Position')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'Time step: {t_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(time_indices), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()

    def create_animation(self, interval=20):
        """Create an animation of the wave packet evolution.
        
        Parameters:
        - interval: Time between frames in milliseconds (default: 20)
        
        Returns:
        Matplotlib animation object.
        """
        Nx, Nt = self.B.shape
        
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        plt.axis([0, Nx, 0, np.max(self.V)*1.1])
        
        # Add title with barrier parameters
        plt.title(f'Quantum Tunneling Animation - Barrier Width: {self.barrier_width}, Barrier Height: {self.barrier_height}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Probability Density / Potential')
        
        # Initialize plot elements
        myline, = plt.plot([], [], 'r', lw=2, label='|ψ|²')
        myline1, = plt.plot(self.x, self.V, 'k', lw=2, 
                           label=f'Barrier (Width={self.barrier_width}, Height={self.barrier_height})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Animation update function
        def animate(i):
            myline.set_data(self.x, np.abs(self.B[:, i]))
            myline1.set_data(self.x, self.V)
            return myline, myline1
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=interval)
        return anim

    def verify_probability_conservation(self):
        """Verify that total probability is conserved during evolution.
        
        Returns:
        Array of total probability at each time step.
        """
        total_prob = np.zeros(self.Nt)
        for t in range(self.Nt):
            total_prob[t] = np.sum(np.abs(self.B[:, t])**2)
        
        return total_prob

    def demonstrate(self):
        """Run a complete demonstration of the quantum tunneling simulation.
        
        Returns:
        Animation object showing the time evolution.
        """
        print("Quantum Tunneling Simulation")
        print("=" * 40)
        
        # Solve the Schrödinger equation
        print("Solving Schrodinger equation...")
        self.solve_schrodinger()
        T, R = self.calculate_coefficients()
        
        # Print results
        print(f"\n势垒宽度:{self.barrier_width}, 势垒高度:{self.barrier_height} 结果")
        print(f"Transmission coefficient: {T:.4f}")
        print(f"Reflection coefficient: {R:.4f}")
        print(f"Total (T + R): {T + R:.4f}")
        
        # Plot evolution
        print("\nPlotting wave function evolution...")
        self.plot_evolution()
        
        # Check probability conservation
        total_prob = self.verify_probability_conservation()
        print(f"\nProbability conservation:")
        print(f"Initial probability: {total_prob[0]:.6f}")
        print(f"Final probability: {total_prob[-1]:.6f}")
        print(f"Relative change: {abs(total_prob[-1] - total_prob[0])/total_prob[0]*100:.4f}%")
        
        # Create animation
        print("\nCreating animation...")
        anim = self.create_animation()
        plt.show()
        
        return anim


def demonstrate_quantum_tunneling():
    """Convenience function to demonstrate quantum tunneling simulation.
    
    Returns:
    Animation object showing the time evolution.
    """
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # Run demonstration with specific barrier parameters
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
