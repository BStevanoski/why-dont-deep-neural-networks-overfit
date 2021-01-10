import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def DGP(x):
    return np.sin(2 * np.pi * x)


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


np.random.seed(2020)
train_size = 12
test_size = 100
train_x = np.linspace(0, 1, train_size)
train_y = np.random.normal(loc=DGP(train_x), scale=0.3)

test_x = np.linspace(0, 1, test_size)
test_y = np.random.normal(loc=DGP(test_x), scale=0.3)

x_grid = np.linspace(0, 1, 100)
y_grid = DGP(x_grid)

train_rmses = []
test_rmses = []
degrees = list(range(0, train_size))
for degree in degrees:
    model = np.poly1d(np.polyfit(train_x, train_y, degree))
    train_rmse = mean_squared_error(train_y, model(train_x), squared=False)
    train_rmses.append(train_rmse)
    test_rmse = mean_squared_error(test_y, model(test_x), squared=False)
    test_rmses.append(test_rmse)

plt.figure()
print(train_rmses)
print(test_rmses)
plt.plot(degrees, train_rmses, 'o-', c='b', label="Train", mfc="none", ms=8)
plt.plot(degrees, test_rmses, 'o-', c='r', label="Test", mfc="none", ms=8)
plt.legend()
plt.xticks(degrees)
plt.xlabel('Polynomial degree')
plt.ylabel('RMSE')
plt.savefig("Images/bias_variance_tradeoff.png", bbox_inches='tight')
# plt.show()

# fig = plt.figure()
# plt.scatter(train_x, train_y, facecolors='none', edgecolors='b')
# plt.plot(x_grid, y_grid, c='g')
# model = np.poly1d(np.polyfit(train_x, train_y, 0))
# predicted_ys = model(x_grid)
# plt.plot(x_grid, predicted_ys, label="degree 0")
# model = np.poly1d(np.polyfit(train_x, train_y, 1))
# predicted_ys = model(x_grid)
# plt.plot(x_grid, predicted_ys, label="degree 1")
# model = np.poly1d(np.polyfit(train_x, train_y, 3))
# predicted_ys = model(x_grid)
# plt.plot(x_grid, predicted_ys, label="degree 3")
# model = np.poly1d(np.polyfit(train_x, train_y, train_size-1))
# predicted_ys = model(x_grid)
# plt.plot(x_grid, predicted_ys, label=f"degree {train_size-1}")
# plt.legend()
# plt.ylim(-1.2, 1.2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

fig = plt.figure()

plt.subplot(221)
plt.title('Degree 0')
plt.scatter(train_x, train_y, facecolors='none', edgecolors='b')
plt.plot(x_grid, y_grid, c='g', label='truth')
model = np.poly1d(np.polyfit(train_x, train_y, 0))
predicted_ys = model(x_grid)
plt.plot(x_grid, predicted_ys, c='r', label='predicted')
plt.ylim(-1.2, 1.2)
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(222)
plt.title('Degree 1')
plt.scatter(train_x, train_y, facecolors='none', edgecolors='b')
plt.plot(x_grid, y_grid, c='g')
model = np.poly1d(np.polyfit(train_x, train_y, 1))
predicted_ys = model(x_grid)
plt.plot(x_grid, predicted_ys, c='r')
plt.ylim(-1.2, 1.2)
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(223)
plt.title('Degree 3')
plt.scatter(train_x, train_y, facecolors='none', edgecolors='b')
plt.plot(x_grid, y_grid, c='g')
model = np.poly1d(np.polyfit(train_x, train_y, 3))
predicted_ys = model(x_grid)
plt.plot(x_grid, predicted_ys, c='r')
plt.ylim(-1.2, 1.2)
plt.xlabel('x')
plt.ylabel('y')

ax = plt.subplot(224)
plt.title(f'Degree {train_size - 1}')
plt.scatter(train_x, train_y, facecolors='none', edgecolors='b')
plt.plot(x_grid, y_grid, c='g', label='truth')
model = np.poly1d(np.polyfit(train_x, train_y, train_size - 1))
predicted_ys = model(x_grid)
plt.plot(x_grid, predicted_ys, c='r', label='predicted')
plt.ylim(-1.2, 1.2)
plt.xlabel('x')
plt.ylabel('y')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left')

plt.tight_layout()
plt.savefig("Images/bias_variance_tradeoff2.png", bbox_inches='tight')
# plt.show()

# root mean squared error


train_rmses = []
test_rmses = []
degrees = list(range(0, 10 * train_size))
for degree in degrees:
    model = np.poly1d(np.polyfit(train_x, train_y, degree))
    train_rmse = mean_squared_error(train_y, model(train_x), squared=False)
    train_rmses.append(train_rmse)
    test_rmse = mean_squared_error(test_y, model(test_x), squared=False)
    test_rmses.append(test_rmse)

plt.figure()
print(train_rmses)
print(test_rmses)
plt.plot(degrees, train_rmses, '-', c='b', label="Train", mfc="none", ms=8)
plt.plot(degrees, test_rmses, '-', c='r', label="Test", mfc="none", ms=8)
plt.legend()
# plt.xticks(degrees)
plt.xlabel('Polynomial degree')
plt.ylabel('RMSE')
plt.savefig("Images/bias_variance_tradeoff3.png", bbox_inches='tight')
