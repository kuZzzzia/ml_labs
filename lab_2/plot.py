import csv
import matplotlib.pyplot as plt

def save_plot(x, arr, path:str):
    fig = plt.figure()
    plt.figure().clear()
    for val, lab in arr:
        plt.plot(x, val, label=lab)
    plt.grid()
    plt.legend()
    plt.savefig(path)

def main():
    with open('ml_labs/lab_2/res/res.csv') as csv_file:
        n = 5
        x_epochs, time, acc, loss = [i for i in range(1, n+1)], [], [], []

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        curr_time = []
        curr_acc = []
        curr_loss = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                curr_time.append(int(row[2]))
                curr_acc.append(float(row[3]))
                curr_loss.append(float(row[4]))
                if line_count % n == 0:
                    time.append((curr_time, row[0]))
                    acc.append((curr_acc, row[0]))
                    loss.append((curr_loss, row[0]))
                    curr_time = []
                    curr_acc = []
                    curr_loss = []
                line_count += 1

        save_plot(x_epochs, time, 'ml_labs/lab_2/res/time.png')
        save_plot(x_epochs, acc, 'ml_labs/lab_2/res/acc.png')
        save_plot(x_epochs, loss, 'ml_labs/lab_2/res/loss.png')

main()