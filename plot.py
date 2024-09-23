import matplotlib.pyplot as plt


def plot_payoff(payoff, title = "Payoff matrix"):
    
    f = plt.figure(1)
    plt.title(title)
    plt.xlabel('Player 2')
    plt.ylabel('Player 1')
    plt.imshow(payoff)
    plt.colorbar()
    f.show()

def plot_potential(potentials):
    
    f = plt.figure(2)
    plt.plot(potentials)
    plt.ylabel('Potential')
    plt.xlabel('Iteration')
    f.show()
    