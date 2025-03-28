import matplotlib.pyplot as plt

def plot_executions(executions):
    times = list(range(len(executions)))
    prices = [e["executed_price"] if "executed_price" in e else e["price"] for e in executions]
    sizes = [e["size"] for e in executions]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Execution Step")
    ax1.set_ylabel("Price", color="tab:blue")
    ax1.plot(times, prices, color="tab:blue", label="Price")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Size", color="tab:green")
    ax2.bar(times, sizes, color="tab:green", alpha=0.3, label="Size")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Execution Price and Size Over Time")
    fig.tight_layout()
    plt.show()