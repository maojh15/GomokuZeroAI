import matplotlib.pyplot as plt
import re

def read_loss_data(filename):
    pattern = r"policy_loss=([\d.]+)\s+value_loss=([\d.]+)\s+total_loss=([\d.]+)"
    pattern = re.compile(pattern)
    policy_losses = []
    value_losses = []
    total_losses = []
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                policy_loss = float(match.group(1))
                value_loss = float(match.group(2))
                total_loss = float(match.group(3))
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                total_losses.append(total_loss)
    return policy_losses, value_losses, total_losses


def plot_loss(policy_losses, value_losses, total_losses):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(policy_losses, label='Policy Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss Over Epochs')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(value_losses, label='Value Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value Loss')
    plt.title('Value Loss Over Epochs')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(total_losses, label='Total Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss Over Epochs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_file_list = [
        r"D:\Projects\MyGames\GomokuZeroAI\result_15x15\log.txt"
    ]
    all_policy_losses = []
    all_value_losses = []
    all_total_losses = []
    for filename in log_file_list:
        policy_losses, value_losses, total_losses = read_loss_data(filename)
        all_policy_losses.extend(policy_losses)
        all_value_losses.extend(value_losses)
        all_total_losses.extend(total_losses)

    plot_loss(all_policy_losses, all_value_losses, all_total_losses)
